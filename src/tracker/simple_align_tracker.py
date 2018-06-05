from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable
import numpy as np
import cv2

class Simple_Align_Tracker():

	def __init__(self, frcnn, detection_person_thresh, regression_person_thresh, detection_nms_thresh,
		regression_nms_thresh, public_detections):
		self.frcnn = frcnn
		self.detection_person_thresh = detection_person_thresh
		self.regression_person_thresh = regression_person_thresh
		self.detection_nms_thresh = detection_nms_thresh
		self.regression_nms_thresh = regression_nms_thresh
		self.public_detections = public_detections

		self.reset()

	def reset(self, hard=True):
		self.ind2track = torch.zeros(0).cuda()
		self.pos = torch.zeros(0).cuda()
		#self.features = torch.zeros(0).cuda()
		#self.kill_counter = torch.zeros(0).cuda()

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def keep(self, keep):
		self.pos = self.pos[keep]
		#self.features = self.features[keep]
		self.ind2track = self.ind2track[keep]
		#self.kill_counter = self.kill_counter[keep]

	def align(self, blob):
		if self.im_index > 0:
			im1 = self.last_image.cpu().numpy()
			im2 = blob['data'][0][0].cpu().numpy()
			im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
			im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
			sz = im1.shape
			warp_mode = cv2.MOTION_EUCLIDEAN
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			#number_of_iterations = 5000
			number_of_iterations = 50
			#number_of_iterations = 10
			#termination_eps = 1e-10
			termination_eps = 0.001
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
			(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
			warp_matrix = torch.from_numpy(warp_matrix)
			pos = []
			for p in self.pos:
				p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
				p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)
				p1_n = torch.mm(warp_matrix, p1).view(1,2)
				p2_n = torch.mm(warp_matrix, p2).view(1,2)
				pos.append(torch.cat((p1_n, p2_n), 1))
			self.pos = torch.cat(pos, 0).cuda()

	def step(self, blob):

		cl = 1

		###########################
		# Look for new detections #
		###########################
		self.frcnn.load_image(blob['data'][0], blob['im_info'][0])
		if self.public_detections:
			if self.public_detections == "DPM_RAW":
				dets = blob['raw_dets']
			elif self.public_detections == "DPM":
				dets = blob['dets']
			else:
				raise NotImplementedError("[!] Public detecions not understood: {}\nChoose between: ['DPM', 'DPM_RAW', False]".format(self.public_detections))
			if len(dets) > 0:
				dets = torch.cat(dets, 0)			
				_, scores, bbox_pred, rois = self.frcnn.test_rois(dets)
			else:
				rois = torch.zeros(0).cuda()
		else:
			_, scores, bbox_pred, rois = self.frcnn.detect()

		if rois.nelement() > 0:
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

			# Filter out tracks that have too low person score
			scores = scores[:,cl]
			inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
		else:
			inds = torch.zeros(0).cuda()

		if inds.nelement() > 0:
			boxes = boxes[inds]
			det_pos = boxes[:,cl*4:(cl+1)*4]
			det_scores = scores[inds]
		else:
			det_pos = torch.zeros(0).cuda()
			det_scores = torch.zeros(0).cuda()

		##################
		# Predict tracks #
		##################
		num_tracks = 0
		nms_inp_reg = self.pos.new(0)
		if self.pos.nelement() > 0:
			# align
			self.align(blob)
			# regress
			_, scores, bbox_pred, rois = self.frcnn.test_rois(self.pos)
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
			#self.pos = boxes[:,cl*4:(cl+1)*4]
			pos = boxes[:,cl*4:(cl+1)*4]

			# get scores of new regressed positions
			#_, scores, _, _ = self.frcnn.test_rois(self.pos)
			scores = scores[:,cl]

			# check if still is a valid person
			#dead = torch.le(scores, self.regression_person_thresh)
			#self.kill_counter[dead] += 1
			#self.kill_counter[~dead] = 0
			#keep = torch.lt(self.kill_counter, self.alive_patience).nonzero()
			keep = torch.gt(scores, self.regression_person_thresh).nonzero()
			if keep.nelement() > 0:
				keep = keep[:,0]
				self.keep(keep)
				scores = scores[keep]
				
				self.pos = pos[keep]

				# create nms input
				#nms_inp_reg = torch.cat((self.pos, self.pos.new(self.pos.size(0),1).fill_(2)),1)
				nms_inp_reg = torch.cat((self.pos, scores.add_(2).view(-1,1)),1)
				#nms_inp_reg = torch.cat((self.pos, torch.rand(scores.size()).add_(2).view(-1,1).cuda()),1)

				# nms here if tracks overlap
				keep = nms(nms_inp_reg, self.regression_nms_thresh)
				self.keep(keep)
				nms_inp_reg = nms_inp_reg[keep]

				# number of active tracks
				num_tracks = nms_inp_reg.size(0)
			else:
				self.reset(hard=False)

		#####################
		# Create new tracks #
		#####################

		# create nms input and nms new detections
		if det_pos.nelement() > 0:
			nms_inp_det = torch.cat((det_pos, det_scores.view(-1,1)), 1)
		else:
			nms_inp_det = torch.zeros(0).cuda()
		if nms_inp_det.nelement() > 0:
			keep = nms(nms_inp_det, self.detection_nms_thresh)
			nms_inp_det = nms_inp_det[keep]
			# check with every track in a single run (problem if tracks delete each other)
			for i in range(num_tracks):
				nms_inp = torch.cat((nms_inp_reg[i].view(1,-1), nms_inp_det), 0)
				keep = nms(nms_inp, self.detection_nms_thresh)
				keep = keep[torch.ge(keep,1)]
				if keep.nelement() == 0:
					nms_inp_det = nms_inp_det.new(0)
					break
				nms_inp_det = nms_inp[keep]

		if nms_inp_det.nelement() > 0:

			num_new = nms_inp_det.size(0)
			new_det_pos = nms_inp_det[:,:4]

			self.pos = torch.cat((self.pos, new_det_pos), 0)

			self.ind2track = torch.cat((self.ind2track, torch.arange(self.track_num, self.track_num+num_new).cuda()), 0)

			self.track_num += num_new

			#self.kill_counter = torch.cat((self.kill_counter, torch.zeros(num_new).cuda()), 0)

		####################
		# Generate Results #
		####################

		for j,t in enumerate(self.pos / blob['im_info'][0][2]):
			track_ind = int(self.ind2track[j])
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			self.results[track_ind][self.im_index] = t.cpu().numpy()

		self.im_index += 1
		self.last_image = blob['data'][0][0]

		#print("tracks active: {}/{}".format(num_tracks, self.track_num))

	def get_results(self):
		return self.results
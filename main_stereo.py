import torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
from pathlib import Path
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from bokeh.models import TabPanel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column, layout, gridplot
from bokeh.models import Div, WheelZoomTool


#### TO LIMIT MEMORY USAGE ####

import resource

soft_limit_in_bytes = 7 * 1024 * 1024 * 1024    # 7 GB
hard_limit_in_bytes = 7 * 1024 * 1024 * 1024    # 7 GB 

resource.setrlimit(resource.RLIMIT_AS, (soft_limit_in_bytes, hard_limit_in_bytes))

###############################


def visualize_paths(gt_path, pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
    output_file(file_out, title=html_tile)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(gt_path - pred_path, axis=1)
    source = ColumnDataSource(data=dict(gtx=gt_path[:, 0], gty=gt_path[:, 1],
                                        px=pred_path[:, 0], py=pred_path[:, 1],
                                        diffx=np.arange(len(diff)), diffy=diff,
                                        disx=xs, disy=ys))

    fig1 = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max", toolbar_location="above",
                  x_axis_label="x", y_axis_label="y")
    fig1.circle("gtx", "gty", source=source, color="blue", hover_fill_color="firebrick", legend_label="GT")
    fig1.line("gtx", "gty", source=source, color="blue", legend_label="GT")

    fig1.circle("px", "py", source=source, color="green", hover_fill_color="firebrick", legend_label="Pred")
    fig1.line("px", "py", source=source, color="green", legend_label="Pred")

    fig1.multi_line("disx", "disy", source=source, legend_label="Error", color="red", line_dash="dashed")
    fig1.legend.click_policy = "hide"

    fig2 = figure(title="Error", tools=tools, width_policy="max", toolbar_location="above",
                  x_axis_label="frame", y_axis_label="error")
    fig2.circle("diffx", "diffy", source=source, hover_fill_color="firebrick", legend_label="Error")
    fig2.line("diffx", "diffy", source=source, legend_label="Error")

    show(layout([Div(text=f"<h1>{title}</h1>"),
                 Div(text="<h2>Paths</h1>"),
                 [fig1, fig2],
                 ], sizing_mode='scale_width'))
    
    
def save_poses(path, poses):
    with open(path, 'a') as fo:
        for j in poses:
            pose3x4 = j #[:3, :]
            flat_array = pose3x4.flatten()
            
            # Accumula i dati in un buffer
            buffer_data = ' '.join(map(str, flat_array)) + ' \n'
            fo.write(buffer_data)
            
        #fo.write('\n')
    print(f"{poses.shape[0]} written to file")

        
            

class VisualOdometry():
    def __init__(self, data_dir, NUM_IMAGES):
        self.N = NUM_IMAGES
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib('calib.txt')
        self.gt_poses = self._load_poses('../poses/00.txt')
        self.im_path_l = self._load_paths(data_dir + 'image_0/', self.N)
        self.im_path_r = self._load_paths(data_dir + 'image_1/', self.N)
        #self.images_l = self._load_images(data_dir + '/image_0', self.N)
        #self.images_r = self._load_images(data_dir + '/image_1', self.N)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.extractor = SuperPoint(num_max_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue("superpoint").eval().to(self.device)

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=block, P1=P1, P2=P2)
        self.disp_prev = np.divide(self.disparity.compute(self._load_images(self.im_path_l[0]), self._load_images(self.im_path_r[0])).astype(np.float32), 16)
        self.disp_cur = None
        #self.disparities = [
            #np.divide(self.disparity.compute(self._load_images(self.im_path_l[0]), self._load_images(self.im_path_r[0])).astype(np.float32), 16)]
    

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_paths(filepath, NUM_IMAGES):
        """
        Loads path of images
        
        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        paths (list): list with ordered images' path
        """

        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        if NUM_IMAGES != 0:
            image_paths = image_paths[:NUM_IMAGES]
          
        return image_paths
    
    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        image (cv2.Mat): grayscale images. Shape (n, height, width)
        """
        images = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        
        return residuals
    
    def get_matches(self, im1, im2):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using SuperPoint algorithm

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
       
        # Find the keypoints and descriptors with SuperPoint
        feats0 = self.extractor.extract(numpy_image_to_torch(im1).to(self.device))
        feats1 = self.extractor.extract(numpy_image_to_torch(im2).to(self.device))      
        
        # Find matches
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        
        # Remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        

        scores = matches01["scores"].detach().numpy()
        
        
        good = []
        for j in range(len(scores)):
            if scores[j] > 0.99:
                good.append(matches[j])
        
        
        if len(good) > 400:
            good = good[:400]
    
        m_kpts0 = kpts0.detach().cpu().numpy()[[good[x][0] for x in range(len(good))]]
        m_kpts1 = kpts1.detach().cpu().numpy()[[good[x][1] for x in range(len(good))]]


        # Get the image points form the good matches
        q1 = np.float64(m_kpts0)
        q2 = np.float64(m_kpts1)
      
        return q1, q2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=5.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
                
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=150):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 7

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0
        th1 = 128

        for n in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]
            
            if n == 0:
                in_guess = np.array([0, 0, 0, 0, 0, -1])
                weighted_residuals = np.ones_like(sample_q1)
                better_sol = False
            else:
                # Compute weights with IRLS
                weights = th1 / max(th1, np.abs(error))#np.sqrt(np.abs(error) + 1e-9)
                weights /= np.sum(weights)
                weighted_residuals = weights * self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
                in_guess = opt_res.x
                
            if better_sol == True:
                in_guess = out_pose
                better_sol = False 
                
           
                
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200, ftol=1e-4, xtol=1e-4, f_scale=weighted_residuals,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
                better_sol = True
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, i, d_mode):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index
        d_mode (boolean): if yes, set the debug mode (show right images)

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        plt.cla()
        
        # Get the i-1'th image and i'th image
        img1_l = self._load_images(self.im_path_l[i - 1])
        img2_l = self._load_images(self.im_path_l[i])#self.images_l[i - 1:i + 1]
        

        # Track the keypoints
        tp1_l, tp2_l = self.get_matches(img1_l,img2_l)

        # Calculate the disparity
        self.disp_cur = np.divide(self.disparity.compute(img2_l, self._load_images(self.im_path_r[i])).astype(np.float32), 16)
        #self.disparities.append(np.divide(self.disparity.compute(img2_l, self._load_images(self.im_path_r[i])).astype(np.float32), 16))

        # Calculate the right keypoints
        #tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disp_prev, self.disp_cur)
        self.disp_prev = self.disp_cur
        
        # Show keypoints left camera
        matches_cv2_l = [cv2.DMatch(j, j, 0) for j in range(len(tp1_l))]
        kp1_l = [cv2.KeyPoint(x=tp1_l[j, 0], y=tp1_l[j, 1], size=10) for j in range(len(tp1_l))]
        kp2_l = [cv2.KeyPoint(x=tp2_l[j, 0], y=tp2_l[j, 1], size=10) for j in range(len(tp2_l))] 
        res_l =cv2.drawMatches(img1_l, kp1_l, img2_l, kp2_l, matches_cv2_l, None, matchColor= (0,255,0), singlePointColor= None, matchesMask=None, flags=2, matchesThickness=1)
        
        if d_mode == True:
            # Show keypoints right camera
            matches_cv2_r = [cv2.DMatch(j, j, 0) for j in range(len(tp1_r))]
            kp1_r = [cv2.KeyPoint(x=tp1_r[j, 0], y=tp1_r[j, 1], size=10) for j in range(len(tp1_r))]
            kp2_r = [cv2.KeyPoint(x=tp2_r[j, 0], y=tp2_r[j, 1], size=10) for j in range(len(tp2_r))] 
            res_r =cv2.drawMatches(self._load_images(self.im_path_r[i-1]), kp1_r, self._load_images(self.im_path_r[i]), kp2_r, matches_cv2_r, None, matchColor= (0,255,0), singlePointColor= None, matchesMask=None, flags=2, matchesThickness=1)
        
        plt.imshow(res_l, aspect=1.5)
        plt.title(f"LEFT CAMERA\nFrame {i} ---> {i+1}")
        
        if d_mode == True:
            plt.figure(2, figsize=(6,2))
            plt.imshow(res_r, aspect=1.5)
            plt.title(f"RIGHT CAMERA\nFrame {i} ---> {i+1}")
        
        plt.pause(0.1)

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        
        
        #np.delete(self.disparities, i)
        #print(f"Disparity array: {len(self.disparities)}")
        
        return transformation_matrix


def main():
    
    start = time.perf_counter() # for debug purpose
    data_dir = '../data_odometry_gray/dataset/sequences/00/' 
    MAX_FRAMES = 1000    #4541
    vo = VisualOdometry(data_dir, MAX_FRAMES)
    debug_mode = False
    
    out_file = "results/00.txt"
    plt.figure(1, figsize=(6,2))

    gt_path = []
    estimated_path = []
    est_poses = np.empty((0,3,4))
        
    for i in range(MAX_FRAMES):
        try:
            print(f"Iteration {i+1} / {MAX_FRAMES}")
            gt_pose = vo.gt_poses[i]
            if i < 1:
                cur_pose = gt_pose
                cur_pose = cur_pose[:3, :]
            else:
                transf = vo.get_pose(i, debug_mode)
                cur_pose = np.matmul(cur_pose, transf)
                cur_pose = cur_pose[:3, :]
                
            if est_poses.size == 0:
                est_poses = np.expand_dims(cur_pose, axis=0)
            else:
                est_poses = np.concatenate([est_poses, np.expand_dims(cur_pose, axis=0)], axis=0)
            
            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
            
            if (i % 10 == 0) or i == MAX_FRAMES -1:    
                save_poses(out_file, est_poses)
                est_poses = np.empty((0,3,4))
            os.system('clear')
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            
    visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry", title= "Stereo VO with SuperPoint + LightGlue ",
                            file_out=os.path.basename(data_dir) + "stereo.html")
    
    end = time.perf_counter() # for debug purpose
    elapsed = round(end - start, 3)
    print(f"Elapsed time: {elapsed}s --- {round(elapsed/60, 3)}m ")


if __name__ == "__main__":
    main()
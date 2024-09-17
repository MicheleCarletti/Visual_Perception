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




class VisualOdometryLightGlue():
    def __init__(self, data_dir, d):
        self.K, self.P = self._load_calib("calib.txt")
        self.gt_poses = self._load_poses("../poses/01.txt")
        self.images = self._load_images(data_dir)
        self.device = d
        self.extractor = SuperPoint(num_max_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue("superpoint").eval().to(self.device)


    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
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
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using SuperPoint algorithm

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        #clear_output(wait=True)
        plt.cla()
        # Find the keypoints and descriptors with SuperPoint
        feats0 = self.extractor.extract(numpy_image_to_torch(self.images[i-1]).to(self.device))
        feats1 = self.extractor.extract(numpy_image_to_torch(self.images[i]).to(self.device))      
        
        # Find matches
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        # Remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        
        
        

        scores = matches01["scores"].detach().numpy()
        #print(len(scores))
        
        good = []
        for j in range(len(scores)):
            if scores[j] > 0.99:
                good.append(matches[j])
        
        
        #print(good)
        #test0, test1 = kpts0[matches[...,0]], kpts1[matches[...,1]]
        #m_kpts0 = kpts0[[good[x][0] for x in range(len(good))]].detach().numpy()
        #m_kpts1 = kpts1[[good[x][1] for x in range(len(good))]].detach().numpy()
        m_kpts0 = kpts0.detach().cpu().numpy()[[good[x][0] for x in range(len(good))]]
        m_kpts1 = kpts1.detach().cpu().numpy()[[good[x][1] for x in range(len(good))]]

        
        matches_cv2 = [cv2.DMatch(j, j, 0) for j in range(len(good))]
        keypoints1 = [cv2.KeyPoint(x=m_kpts0[j, 0], y=m_kpts0[j, 1], size=10) for j in range(len(m_kpts0))]
        keypoints2 = [cv2.KeyPoint(x=m_kpts1[j, 0], y=m_kpts1[j, 1], size=10) for j in range(len(m_kpts1))] 
        res =cv2.drawMatches( self.images[i-1], keypoints1, self.images[i], keypoints2, matches_cv2, None, matchColor= (0,255,0), singlePointColor= None, matchesMask=None, flags=2, matchesThickness=1)
      
        if i > 0:
            #plt.figure(figsize=(1,1))
            plt.imshow(res, aspect=1.5)
            plt.title(f"Frame {i+1}")
            plt.pause(0.1)
            #plt.clf()
        #cv2.imshow("image", res)
        #cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float64(m_kpts0)
        q2 = np.float64(m_kpts1)
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # DEBUG: to prevent overflow, normalize t 
            t /= np.linalg.norm(t)
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]
    
    def optimize_LM(self, q1, q2, transf):
        R = transf[:3, :3]
        t = transf[:3, 3]
        
        P = np.matmul(np.concatenate((self.K, np.zeros((3,1))), axis=1), transf)
        
        h_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
        h_Q2 = np.matmul(transf, h_Q1)
        
        Q1 = h_Q1[:3, :] / h_Q1[3, :]
        Q2 = h_Q2[:3, :] / h_Q2[3, :]
        
        Q1 = Q1.T
        Q2 = Q2.T
        
        def residuals(p, q1, q2, Q1, Q2):
            """
            transf = p.reshape((4,4))
            R = transf[:3, :3]
            t = transf[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            proj_q2_cv2, _ = cv2.projectPoints(p3d, rvec, t, self.K, None)
            proj_q2 = np.squeeze(proj_q2_cv2)
            #print(f"Projection : {proj_q2.shape}")
            #print(f"Real points : {q2.shape}")
            reprojection_error = np.abs(proj_q2.flatten() - q2.flatten())
            """
            #print(f"q1 {q1.shape}")
            #print(f"q2 {q2.shape}")
            #print(f"Q1 {Q1.shape}")
            #print(f"Q2 {Q2.shape}")
            r = p[:3]
            R, _ = cv2.Rodrigues(r)
            t = p[3:]
            transf = self._form_transf(R,t)
            
            f_projection = np.matmul(self.P, transf)
            b_projection = np.matmul(self.P, np.linalg.inv(transf))
            
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
            residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
            return residuals
        
        early_term_th = 5
        max_iter = 100
        min_error = float('inf')
        early_term = 0
        
        for n in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]
            
            if n == 0:
                initial_guess = np.array([0, 0, 0, 0, 0, -1])#transf[:3, :].flatten()
                better_sol = False
            else:
                if better_sol == True:
                    initial_guess = out_pose
                    better_sol = False
                else:
                    initial_guess = result.x
            #print(f"InG: {initial_guess.shape}")
            result = least_squares(residuals, initial_guess, args=(sample_q1, sample_q2, sample_Q1, sample_Q2), method='lm', max_nfev=200)
            # Calculate the error for the optimized transformation
            error = residuals(result.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))
            
            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = result.x
                early_term = 0
                better_sol = True
            else:
                early_term += 1
            if early_term == early_term_th:
                # If we have not fund any better result in early_termi_th iterations
                break
            
        r_o = out_pose[:3]
        R_o, _ = cv2.Rodrigues(r_o)
        t_o = out_pose[3:]
        optimized_transf = self._form_transf(R_o, t_o)
        #print(f"OT:\n {optimized_transf}")
        
        return optimized_transf
    
    def optimize_IRLS(self, q1, q2, transf):
        R = transf[:3, :3]
        t = transf[:3, 3]
        
        P = np.matmul(np.concatenate((self.K, np.zeros((3,1))), axis=1), transf)
        
        h_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
        h_Q2 = np.matmul(transf, h_Q1)
        
        Q1 = h_Q1[:3, :] / h_Q1[3, :]
        Q2 = h_Q2[:3, :] / h_Q2[3, :]
        
        Q1 = Q1.T
        Q2 = Q2.T
        
        def residuals(p, q1, q2, Q1, Q2):
            """
            transf = p.reshape((4,4))
            R = transf[:3, :3]
            t = transf[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            proj_q2_cv2, _ = cv2.projectPoints(p3d, rvec, t, self.K, None)
            proj_q2 = np.squeeze(proj_q2_cv2)
            #print(f"Projection : {proj_q2.shape}")
            #print(f"Real points : {q2.shape}")
            reprojection_error = np.abs(proj_q2.flatten() - q2.flatten())
            """
            #print(f"q1 {q1.shape}")
            #print(f"q2 {q2.shape}")
            #print(f"Q1 {Q1.shape}")
            #print(f"Q2 {Q2.shape}")
            r = p[:3]
            R, _ = cv2.Rodrigues(r)
            t = p[3:]
            transf = self._form_transf(R,t)
            
            f_projection = np.matmul(self.P, transf)
            b_projection = np.matmul(self.P, np.linalg.inv(transf))
            
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
            residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
            return residuals
        
        early_term_th = 5
        max_iter = 100
        min_error = float('inf')
        early_term = 0
        
        for n in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]
            
            if n == 0:
                initial_guess = np.array([0, 0, 0, 0, 0, -1])#transf[:3, :].flatten()
                weighted_residuals = np.ones_like(sample_q1)
                better_sol = False
            
            else:
                # Compute weights with IRLS
                weights = 1/ np.sqrt(np.abs(error) + 1e-9)
                weights /= np.sum(weights)
                weighted_residuals = weights * residuals(result.x, q1, q2, Q1, Q2)
                initial_guess = result.x
            
            if better_sol == True:
                initial_guess = out_pose
                better_sol = False
                
            #print(f"InG: {initial_guess.shape}")
            result = least_squares(residuals, initial_guess, method='lm', max_nfev=200, ftol=1e-4, xtol=1e-4, f_scale=weighted_residuals,
                                       args=(sample_q1, sample_q2, sample_Q1, sample_Q2))
            # Calculate the error for the optimized transformation
            error = residuals(result.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))           
            
            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = result.x
                early_term = 0
                better_sol = True
            else:
                early_term += 1
            if early_term == early_term_th:
                # If we have not fund any better result in early_termi_th iterations
                break
            
        r_o = out_pose[:3]
        R_o, _ = cv2.Rodrigues(r_o)
        t_o = out_pose[3:]
        optimized_transf = self._form_transf(R_o, t_o)
        #print(f"OT:\n {optimized_transf}")
        
        return optimized_transf
    
    def remove_outliers(self, q1, q2, th = 2.0):
        """
        Rimuove gli outliers dagli array q1 e q2 basandosi sulla deviazione standard della distanza euclidea.

        Parameters
        ----------
        q1 (ndarray): Array di punti nell'immagine i-1
        q2 (ndarray): Array di punti corrispondenti nell'immagine i
        threshold (float): Soglia in termini di deviazione standard per considerare un punto come outlier

        Returns
        -------
        q1_filtered (ndarray): Array di punti senza outliers nell'immagine i-1
        q2_filtered (ndarray): Array di punti corrispondenti senza outliers nell'immagine i
        """
        # Calcola la distanza euclidea tra i punti corrispondenti
        distances = np.linalg.norm(q1 - q2, axis=1)
        
        # Calcola la deviazione standard delle distanze
        std_dev = np.std(distances)
        
        # Filtra gli outliers basandosi sulla deviazione standard
        mask = distances < (th * std_dev)
        
        # Applica la maschera per rimuovere gli outliers
        q1_filtered = q1[mask]
        q2_filtered = q2[mask]
        
        print(f"Mask: {mask}")
        print(f"q1 before filter: {q1.shape}")
        print(f"q2 before filter: {q2.shape}")
        print(f"q1 after filter: {q1_filtered.shape}")
        print(f"q2 after filter: {q2_filtered.shape}")
        
        if (len(q1_filtered)< 10 or len(q2_filtered) < 10):
            return q1, q2
        else:
            return q1_filtered, q2_filtered

def main():
    
    start = time.perf_counter() # for debug purpose
    images = Path("../data_odometry_gray/dataset/sequences/01/image_0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vo = VisualOdometryLightGlue(images, device)
    plt.figure(figsize=(6,2))
    debug_mode = True

    #play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i in range(200):
        gt_pose = vo.gt_poses[i]
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            # Rimuovi gli outliers
            #q1, q2 = vo.remove_outliers(q1, q2)
            transf = vo.get_pose(q1, q2)
            opt_transf = vo.optimize_IRLS(q1, q2, transf)
            opt_transf = np.nan_to_num(opt_transf, neginf=0, posinf=0) #DEBUG: avoid overflow or underflow
            #cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            cur_pose = np.matmul(cur_pose, opt_transf)
            if debug_mode:
                print(f"\nFrame {i+1}")
                print(f"Estimated pose:\n {cur_pose}")
                print(f"Real pose:\n {gt_pose}")
                #print(f"Optimized pose:\n {optim_pose}")
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    visualize_paths(gt_path, estimated_path, "Visual Odometry", title="Mono VO SuperPoint + LightGlue with local optim", file_out="mono.html")
    end = time.perf_counter() # for debug purpose
    elapsed = round(end - start, 3)
    print(f"Elapsed time: {elapsed}s --- {round(elapsed/60, 3)}m ")

if __name__ == "__main__":
    main()
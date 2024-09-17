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
        self.gt_poses = self._load_poses("../poses/00.txt")
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
            if scores[j] > 0.95:
                good.append(matches[j])
        
        
        #print(good)
        #test0, test1 = kpts0[matches[...,0]], kpts1[matches[...,1]]
        m_kpts0 = kpts0[[good[x][0] for x in range(len(good))]].detach().numpy()
        m_kpts1 = kpts1[[good[x][1] for x in range(len(good))]].detach().numpy()

        
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

def main():
    
    start = time.perf_counter() # for debug purpose
    images = Path("../data_odometry_gray/dataset/sequences/00/image_0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vo = VisualOdometryLightGlue(images, device)
    plt.figure(figsize=(8,3))
    debug_mode = True

    #play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i in range(100):
        gt_pose = vo.gt_poses[i]
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            transf = np.nan_to_num(transf, neginf=0, posinf=0) #DEBUG: avoid overflow or underflow
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            if debug_mode:
                print(f"\nFrame {i+1}")
                print(f"Estimated pose:\n {cur_pose}")
                print(f"Real pose:\n {gt_pose}")
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    visualize_paths(gt_path, estimated_path, "Visual Odometry", title="VO SuperPoint + LightGlue no local optim", file_out=os.path.basename(".") + ".html")
    end = time.perf_counter() # for debug purpose
    elapsed = round(end - start, 3)
    print(f"Elapsed time: {elapsed}s --- {round(elapsed/60, 3)}m ")

if __name__ == "__main__":
    main()
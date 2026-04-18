"""
Q2.3.1: SIFT keypoint detection, matching, RANSAC inliers, and visualization.
"""
import os

import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform


def poi_detection(im1, im2, save_npz_path=None, visualization_path=None):
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) if im1.ndim == 3 else im1
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) if im2.ndim == 3 else im2

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    empty = (
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    )
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        if save_npz_path is not None:
            np.savez(save_npz_path, pts1=empty[0], pts2=empty[1])
        return empty

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_pairs = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn_pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) == 0:
        if save_npz_path is not None:
            np.savez(save_npz_path, pts1=empty[0], pts2=empty[1])
        return empty

    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])

    if len(good) < 4:
        pts1, pts2 = src.copy(), dst.copy()
        inlier_matches = list(good)
    else:
        _, inliers = ransac(
            (src, dst),
            AffineTransform,
            min_samples=4,
            residual_threshold=8.0,
            max_trials=10000,
        )
        if inliers is None:
            pts1, pts2 = src.copy(), dst.copy()
            inlier_matches = list(good)
        else:
            pts1 = src[inliers]
            pts2 = dst[inliers]
            inlier_matches = [good[i] for i in range(len(good)) if inliers[i]]

    inlier_matches.sort(key=lambda m: m.distance)
    top100 = inlier_matches[: min(100, len(inlier_matches))]

    if visualization_path is not None:

        matchpane = cv2.drawMatches(
            im1,
            kp1,
            im2,
            kp2,
            top100,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
        )
        vis = matchpane
        cv2.imwrite(visualization_path, vis)

    if save_npz_path is not None:
        np.savez(save_npz_path, pts1=pts1.astype(np.float32), pts2=pts2.astype(np.float32))

    return pts1, pts2


if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    default_npz = os.path.join(_dir, "q2.3_1.npz")
    default_vis = os.path.join(_dir, "q2.3_1_visualization.png")

    img1 = cv2.imread("data/image1.jpg")
    img2 = cv2.imread("data/image2.jpg")
    if img1 is None or img2 is None:
        raise SystemExit(f"Could not read images: {img1!r}, {img2!r}")

    pts1, pts2 = poi_detection(
        img1,
        img2,
        save_npz_path=default_npz,
        visualization_path=default_vis,
    )
    print(f"Saved {len(pts1)} matches to {default_npz}")
    print(f"Visualization: {default_vis}")

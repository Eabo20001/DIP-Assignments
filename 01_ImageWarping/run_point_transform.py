import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

def mls_affine_batch(v, source_pts, target_pts, alpha=1.0, epsilon=1e-8):
    """
    Compute the new positions of multiple points v under affine MLS deformation.

    Parameters:
        v : array_like, shape (M,2) or (2,)
            Coordinates of points to be deformed
        source_pts : array_like, shape (N,2)
            Source control points
        target_pts : array_like, shape (N,2)
            Target control points
        alpha : float, default 1.0
            Weight decay parameter, weight = 1 / (distance^(2*alpha))
        epsilon : float, default 1e-8
            Threshold for considering a point coincident with a source point

    Returns:
        numpy.ndarray, shape (M,2)
            Deformed coordinates for each input point
    """
    # Convert to numpy arrays and ensure correct shape
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    p = np.asarray(source_pts, dtype=float).reshape(-1, 2)
    q = np.asarray(target_pts, dtype=float).reshape(-1, 2)

    M, N = v.shape[0], p.shape[0]

    # If no control points, return v directly
    if N == 0:
        return v.copy()

    # ---------- Step 1: Compute squared distances from all v to all source points ----------
    # diff[i,j] = v_i - p_j, shape (M, N, 2)
    diff = v[:, None, :] - p[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)   # (M, N)

    # ---------- Step 2: Handle v that coincide with source points ----------
    min_dist_sq = np.min(dist_sq, axis=1)
    fixed_mask = min_dist_sq < epsilon      # points that need direct mapping
    fixed_indices = np.argmin(dist_sq, axis=1)[fixed_mask]

    # Initialize result array
    result = np.zeros((M, 2), dtype=float)

    if np.any(fixed_mask):
        result[fixed_mask] = q[fixed_indices]

    # Remaining points (non-coincident) need deformation
    remain_mask = ~fixed_mask
    if not np.any(remain_mask):
        return result

    v_rem = v[remain_mask]                  # (M_rem, 2)
    dist_sq_rem = dist_sq[remain_mask, :]   # (M_rem, N)

    # ---------- Step 3: Compute weights ----------
    # Avoid division by zero (but coincident points have been excluded, distances > epsilon here)
    w = 1.0 / (dist_sq_rem ** alpha)        # (M_rem, N)

    # ---------- Step 4: Compute weighted centroids ----------
    sum_w = np.sum(w, axis=1, keepdims=True)   # (M_rem, 1)
    p_star = (w @ p) / sum_w                    # (M_rem, 2)
    q_star = (w @ q) / sum_w

    # ---------- Step 5: Construct relative coordinates ----------
    # p_hat[i,j] = p[j] - p_star[i], shape (M_rem, N, 2)
    p_hat = p[None, :, :] - p_star[:, None, :]
    q_hat = q[None, :, :] - q_star[:, None, :]

    # ---------- Step 6: Compute A and B matrices for each v ----------
    # A_i = sum_j w_ij * (p_hat_ij^T * p_hat_ij)   (2x2)
    # B_i = sum_j w_ij * (p_hat_ij^T * q_hat_ij)   (2x2)
    # Efficient computation using einsum
    # w_ij: (M_rem,N); p_hat: (M_rem,N,2); q_hat: (M_rem,N,2)
    A = np.einsum('ij, ijk, ijl -> ikl', w, p_hat, p_hat)   # (M_rem,2,2)
    B = np.einsum('ij, ijk, ijl -> ikl', w, p_hat, q_hat)

    # ---------- Step 7: Solve linear system A @ M = B ----------
    # Batch solve the linear systems (M_rem,2,2)
    try:
        # Try using solve, might encounter singular matrices
        M_mat = np.linalg.solve(A, B)          # (M_rem,2,2)
    except np.linalg.LinAlgError:
        # If singular, fallback to pseudoinverse for each singular point (slower, for robustness only)
        M_mat = np.zeros_like(A)
        for i in range(A.shape[0]):
            try:
                M_mat[i] = np.linalg.solve(A[i], B[i])
            except np.linalg.LinAlgError:
                M_mat[i] = np.linalg.pinv(A[i]) @ B[i]

    # ---------- Step 8: Compute deformed points ----------
    v_diff = v_rem - p_star                    # (M_rem,2)
    # Multiply v_diff with M_mat: v_diff @ M_mat, resulting in (M_rem,2)
    f_rem = np.einsum('ij, ijk -> ik', v_diff, M_mat) + q_star

    # Fill back the results
    result[remain_mask] = f_rem

    return result

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """
    if image is None:
        return None

    warped_image = np.array(image)

    source_pts = np.asarray(source_pts, dtype=float).reshape(-1, 2)
    target_pts = np.asarray(target_pts, dtype=float).reshape(-1, 2)

    # MLS needs matched source/target pairs. If there are not enough pairs to
    # define a deformation, keep the original image.
    num_pairs = min(len(source_pts), len(target_pts))
    if num_pairs == 0:
        return warped_image

    source_pts = source_pts[:num_pairs]
    target_pts = target_pts[:num_pairs]

    h, w = warped_image.shape[:2]

    # Build a dense grid over the output image. We perform inverse warping:
    # for each output pixel, estimate where it came from in the input image by
    # mapping target control points back to source control points.
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    output_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    input_coords = mls_affine_batch(
        output_coords,
        target_pts,
        source_pts,
        alpha=alpha,
        epsilon=eps,
    )

    map_x = input_coords[:, 0].reshape(h, w).astype(np.float32)
    map_y = input_coords[:, 1].reshape(h, w).astype(np.float32)

    warped_image = cv2.remap(
        warped_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch(server_name="0.0.0.0", server_port=7855)

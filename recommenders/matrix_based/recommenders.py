import numpy as np

def recommend_bpr_user_index(user_matrix, item_matrix, n_recommendations, user_index):
    user_vector = user_matrix[user_index, :]
    scoring = np.matmul(user_vector, item_matrix.T)
    recommendations = list(np.argsort(scoring))[::-1]
    return recommendations[:n_recommendations]

def recommend_transe_user_index(node_matrix, relation_vector, n_recommendations, user_index, n_users=6863):
    item_affinity = node_matrix[user_index, :] + relation_vector
    item_matrix = node_matrix[n_users:, :]
    scoring = np.linalg.norm(item_affinity - item_matrix, axis=1)
    recommendations = list(np.argsort(scoring))
    return recommendations[:n_recommendations]

def recommend_rotate_user_index(node_matrix, relation_vector, n_recommendations, user_index, n_users=6863):
    real_node_matrix = node_matrix[:, :node_matrix.shape[1] // 2]
    im_node_matrix = node_matrix[:, node_matrix.shape[1] // 2:]
    rel_re, rel_im = np.cos(relation_vector), np.sin(relation_vector)
    re_affinity = real_node_matrix[user_index, :]*rel_re - im_node_matrix[user_index, :]*rel_im - real_node_matrix[n_users:, :]
    im_affinity = im_node_matrix[user_index, :]*rel_re + real_node_matrix[user_index, :]*rel_im - im_node_matrix[n_users:, :]
    scoring = np.linalg.norm(np.stack([re_affinity, im_affinity], axis=2), axis=(1, 2))
    recommendations = list(np.argsort(scoring))
    return recommendations[:n_recommendations]
        
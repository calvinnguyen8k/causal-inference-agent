import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt

def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """
    Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using Augmented Lagrangian Method.
    
    Based on the original NOTEARS implementation:
    Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018).
    DAGs with NO TEARS: Continuous Optimization for Structure Learning.
    Advances in Neural Information Processing Systems, 31.
    """
    def _loss(W):
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - X.shape[1]
        # derivative of tr(exp(W * W)) with respect to W
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into w^+ and w^-
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
            
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

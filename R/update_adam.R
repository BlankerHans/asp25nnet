update_adam <- function(m, v, grad, beta1, beta2, t_global, lr, eps) {
  m_new <- beta1 * m + (1 - beta1) * grad
  v_new <- beta2 * v + (1 - beta2) * (grad^2)
  m_hat <- m_new / (1 - beta1^t_global)
  v_hat <- v_new / (1 - beta2^t_global)
  list(m = m_new, v = v_new, delta = lr * m_hat / (sqrt(v_hat) + eps))
}

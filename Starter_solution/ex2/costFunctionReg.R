costFunctionReg  <- function(X, y, lambda) {
  #COSTFUNCTIONREG Compute cost for logistic regression with regularization
  #   J <- COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  #   theta as the parameter for regularized logistic regression
  function(theta)  {
    # Initialize some useful values
    m <- length(y) # number of training examples
    
    # You need to return the following variables correctly
    J <- 0
    # ----------------------- YOUR CODE HERE -----------------------
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    
    h <- sigmoid(X %*% theta)
    
    costFunction <- (1 / m) * (-1 * t(y) %*% log(h) - t(1 - y) %*% log(1 - h))
    
    # NOTE: the bias term, theta_0, is excluded.
    regTerm <- (lambda / (2 * m)) * t(theta[-1]) %*% theta[-1]
    
    costFunction + regTerm
    
    # ----------------------------------------------------
  }
}

gradReg  <- function (X, y, lambda) {
  #COSTFUNCTIONREG Compute gradient for logistic regression with regularization
  #   J <- COSTFUNCTIONREG(theta, X, y, lambda) computes the
  #   gradient of the cost w.r.t. to the parameters.
  function(theta)  {
    # Initialize some useful values
    m <- length(y) # number of training examples
    
    # You need to return the following variables correctly
    grad <- rep(0,length(theta))
    
    # ----------------------- YOUR CODE HERE -----------------------
    # Instructions: Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    
    h <- sigmoid(X %*% theta)
    
    grad <- (1 / m) * t(X) %*% (h - y)
    
    # NOTE: the bias term, theta_0, is excluded.
    regTerm <- (lambda / m) * c(0, theta[-1])
    
    grad + regTerm
    
    # ----------------------------------------------------
  }
}

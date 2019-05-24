nnCostFunction  <-
  function(input_layer_size, hidden_layer_size, num_labels,X, y, lambda) {
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    #   J <- NNCOSTFUNCTON(hidden_layer_size, num_labels, ...
    #   X, y, lambda)(nn_params) computes the cost of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    
    function(nn_params) {
      # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
      # for our 2 layer neural network
      Theta1 <-
        matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
               hidden_layer_size, (input_layer_size + 1))
      
      Theta2 <-
        matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
               num_labels, (hidden_layer_size + 1))
      
      # Setup some useful variables
      m <- dim(X)[1]
      
      # You need to return the following variable correctly
      J <- 0
      
      # ----------------------- YOUR CODE HERE -----------------------
      # Instructions: You should complete the code by working through the
      #               following parts.
      #
      # Part 1: Feedforward the neural network and return the cost in the
      #         variable J. After implementing Part 1, you can verify that your
      #         cost function computation is correct by verifying the cost
      #         computed in ex4.R
      #
      # Part 2: Implement regularization with the cost function.
      #
      #
    
      # y is a vector with the true class numbers that should be associated
      #  with the rows in X.
      # The output of the neural net, a3, is a matrix with dimensions 
      #  (rows X) x num_classes.  Consider an example of the first row in case
      #  num_classes = 3:
      #   X = [0.56, 0.23, 0.76], a3 = [0.87, 0.32, 0.01], y = 1
      # To calculate the error in a3, we need to put the true answer in the same
      #  form: y -> [1, 0, 0]. That is, y -> Y, dim(Y) = dim(a3). 
    
      # y -> Y
      I <- diag(num_labels)
      Y <- matrix(0, m, num_labels)
      
      for (i in 1:m) {
        true_vec <- I[y[i], ]
        Y[i, ] <- true_vec
      }
      
      # forward feed
      a1 <- cbind(1, X)
      z2 <- Theta1 %*% t(a1)
      z2 <- t(z2)
      a2 <- cbind(1, sigmoid(z2))
      z3 <- Theta2 %*% t(a2)
      z3 <- t(z3)
      a3 <- sigmoid(z3)
      
      # calculate cost
      cost <- sum(-Y * log(a3) - (1 - Y) * log(1 - a3)) / m
      
      # add regularization -- do not regularize the terms that correspond to
      # the bias (i.e. the first column of each matrix)
      reg <- lambda * (sum(Theta1[, -1] ^ 2) + sum(Theta2[, -1] ^ 2)) / (2 * m)
      
      cost + reg
    }
  }

nnGradFunction  <-
  function(input_layer_size, hidden_layer_size, num_labels,
           X, y, lambda) {
    #nnGradFunction Implements the neural network gradient function for a two layer
    #neural network which performs classification
    #   grad <- nnGradFunction(hidden_layer_size, num_labels, ...
    #   X, y, lambda)(nn_params) computes the gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #
    function(nn_params) {
      # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
      # for our 2 layer neural network
      Theta1 <-
        matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
               hidden_layer_size, (input_layer_size + 1))
      
      Theta2 <-
        matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
               num_labels, (hidden_layer_size + 1))
      
      # Setup some useful variables
      m <- dim(X)[1]
      
      # You need to return the following variables correctly
      Theta1_grad <- matrix(0,dim(Theta1)[1],dim(Theta1)[2])
      Theta2_grad <- matrix(0,dim(Theta2)[1],dim(Theta2)[2])
      
      # ----------------------- YOUR CODE HERE -----------------------
      # Instructions: You should complete the code by working through the
      #               following parts.
      #
      # Part 1: Feedforward the neural network
      #
      # Part 2: Implement the backpropagation algorithm to compute the gradients
      #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
      #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
      #         Theta2_grad, respectively. After implementing Part 2, you can check
      #         that your implementation is correct by running checkNNGradients
      #
      #         Note: The vector y passed into the function is a vector of labels
      #               containing values from 1..K. You need to map this vector into a
      #               binary vector of 1's and 0's to be used with the neural network
      #               cost function.
      #
      #         Hint: We recommend implementing backpropagation using a for-loop
      #               over the training examples if you are implementing it for the
      #               first time.
      #
      # Part 3: Implement regularization with the gradients.
      #
      #         Hint: You can implement this around the code for
      #               backpropagation. That is, you can compute the gradients for
      #               the regularization separately and then add them to Theta1_grad
      #               and Theta2_grad from Part 2.
      #
      I <- diag(num_labels)
      Y = matrix(0, m, num_labels)
      for (i in 1:m) {
        Y[i, ] = I[y[i], ]
      }
      
      # Set a1 = x
      a1 <- cbind(1, X)
      # Perform forward propogation to compute al for l = 2:3 = num_layers.
      z2 <- Theta1 %*% t(a1)
      z2 <- t(z2)
      a2 <- cbind(1, sigmoid(z2))
      z3 <- Theta2 %*% t(a2)
      z3 <- t(z3)
      a3 <- sigmoid(z3)
      # Using y, compute d3 = a3 - y
      d3 <- a3 - Y
      # Compute d2 using di = (t(Theta_i)d_{i+1}) .* sigmoidGradient(ai)
      # Note: dim(d3) = 5000 x 10, but the above calculation assumes we are looking
      #       at a single 10 x 1 column vector for one d_3 value. The same orientation
      #       problem is found with a2. To vectorize, do
      #         (t(Theta_i) %*% t(d_{i+1})) .* sigmoidGradient(t(ai))
      #       Since t(AB) = t(B)t(A), t(Theta_i)t(d_{i+1}) = d_{i+1}Theta_i, so
      #       the following is equivalent:
      #         (d_{i+1} %*% Theta_i) .* sigmoidGradient(ai)
      d2 <- (d3 %*% Theta2) * sigmoidGradient(cbind(1, z2))
      # Accumulate the gradient using Di = Di + d_{i+1}t(ai). Note that you should
      # skip or remove the first column of d2 (the column corresponding to the
      # bias term).
      d2 <- d2[ , -1]
      
      D1 <- t(d2) %*% a1
      D2 <- t(d3) %*% a2
      
      # Calculate the Theta gradients, including regularization.
      Theta1_grad <- (D1 + lambda * cbind(0, Theta1[, -1])) / m
      Theta2_grad <- (D2 + lambda * cbind(0, Theta2[, -1])) / m
      
      return(c(c(Theta1_grad), c(Theta2_grad)))
      list(Theta1_grad = D1, Theta2_grad = D2)
      # -------------------------------------------------------------
    }
  }
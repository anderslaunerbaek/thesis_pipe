
#' decode_ctc
#'
#' @param model The Keras model pointer
#' @param images An array with cropped images of the LP and VIN
#'
#' @importFrom keras k_get_value k_ctc_decode k_variable
#' @importFrom dplyr bind_rows
#'
#' @return A tibble with the coresponding decoded text and probability to each of the images.
#'
decode_ctc <- function(model, images) {
  # predict the softmax tensor
  pred <- predict(model, images)

  # decode softmax from softmax tensor
  out_best <- k_get_value(k_ctc_decode(k_variable(pred, dtype = "float32"),
                                       k_variable(array(dim(pred)[2],
                                                        dim = c(dim(pred)[1])),
                                                  dtype = "int32"))[[1]][[1]])

  # get the log probabilities of each decoded sequence
  log_prob <- k_get_value(k_ctc_decode(k_variable(pred, dtype = "float32"),
                                       k_variable(array(dim(pred)[2],
                                                        dim = c(dim(pred)[1])),
                                                  dtype = "int32"))[[2]])
  # conversion
  prob <- exp(-log_prob)

  # create and return a tibble with the coresponding decoded text and probability to each of the images
  bind_rows(lapply(1:dim(pred)[1], function(i) {

    # -1 is the blank symbol from the k_ctc_decode
    x = out_best[i,out_best[i,] != -1]

    # convert from integer to characters and return text and its probability
    list("decoded_text" = num_to_char(x), "prob" = prob[i])
  }))
}



#' model_CRNN
#' inspired by: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
#' and https://github.com/DeepSystems/supervisely-tutorials/blob/master/anpr_ocr/src/image_ocr.ipynb
#'
#' The CRNN model
#'
#' @param no_classes an integer, number of classes.
#' @param batch_shape a list with `batch_shape`.
#' @param glo_set a list must be `glo_set`.
#' @importFrom keras layer_input layer_conv_2d layer_max_pooling_2d
#' @importFrom keras layer_dense layer_add layer_reshape layer_gru
#' @importFrom keras layer_concatenate layer_activation layer_dropout layer_lambda
#' @importFrom keras keras_model k_ctc_batch_cost
#' @importFrom keras compile "%>%" optimizer_adam
#'
#' @return Keras model object
#'
model_CNN_RNN <- function(vocab_size,
                          batch_shape,
                          max_str_len,
                          lr,
                          beta_1,
                          beta_2) {
  # initial values
  conv_filters <- 16
  kernel_size <- c(3, 3)
  p_size <- c(2, 2)
  time_dense_size <- 32
  rnn_cells <- 256
  act <- "relu"

  conv_to_rnn_dims <- c(floor(batch_shape[[2]] / (p_size[1] ^ 2)),
                       floor(batch_shape[[3]] / (p_size[1] ^ 2)) * conv_filters)

  # create model
  input_layer <- layer_input(name = "input", shape = unlist(batch_shape)[-1])
  conv1_layer <- layer_conv_2d(object = input_layer,
                               filters = conv_filters,
                               kernel_size = kernel_size,
                               padding = "same",
                               activation = act,
                               kernel_initializer = "he_normal",
                               name = "conv1")
  max1_layer <- layer_max_pooling_2d(object = conv1_layer,
                                     pool_size = p_size,
                                     name = "max1")
  conv2_layer <- layer_conv_2d(object = max1_layer,
                               filters = conv_filters,
                               kernel_size = kernel_size,
                               padding = "same",
                               activation = act,
                               kernel_initializer = "he_normal",
                               name = "conv2")
  max2_layer <- layer_max_pooling_2d(object = conv2_layer,
                                     pool_size = p_size,
                                     name = "max2")
  #summary(keras_model(input_layer, max2_layer))

  resph_layer <- layer_reshape(object = max2_layer,
                               target_shape = conv_to_rnn_dims,
                               name = "reshape")

  dense_layer <- layer_dense(object = resph_layer,
                             units = time_dense_size,
                             activation = act,
                             kernel_initializer = "he_normal",
                             name = "dense")
  # summary(keras_model(input_layer, dense_layer))

  # create node for first bi-directional LSTM layers
  gru_1a <- layer_gru(object = dense_layer, units = rnn_cells,
                      name = "gru_1a",
                      return_sequences = TRUE,
                      go_backwards = TRUE,
                      kernel_initializer = "he_normal")
  gru_1b <- layer_gru(object = dense_layer, units = rnn_cells,
                      name = "gru_1b",
                      return_sequences = TRUE,
                      go_backwards = TRUE,
                      kernel_initializer = "he_normal")

  #
  gru_1merged <- layer_add(inputs = list(gru_1a, gru_1b), name = "add_gru")
  # summary(keras_model(input_layer, gru_1merged))

  gru_2a <- layer_gru(object = gru_1merged, units = rnn_cells, name = "gru_2a",
                      return_sequences = TRUE,
                      go_backwards = TRUE,
                      kernel_initializer = "he_normal")

  gru_2b <- layer_gru(object = gru_1merged, units = rnn_cells, name = "gru_2b",
                      return_sequences = TRUE,
                      go_backwards = TRUE,
                      kernel_initializer = "he_normal")

  concat_layer <- layer_concatenate(list(gru_2a, gru_2b), name = "concatenate_gru") %>%
    layer_dropout(rate = 0.5, name = "drop")

  # RNN outputs to char activations
  softmax <- layer_dense(object = concat_layer,
                         units = vocab_size,
                         activation = "relu",
                         name = "dense_2",
                         kernel_initializer = "he_normal") %>%
    layer_activation(activation = "softmax", name = "softmax")
  # summary(keras_model(input_layer, softmax))

  # custom CTC loss
  labels <- layer_input(name = "labels", shape = max_str_len, dtype = "float32")
  input_length <- layer_input(name = "input_length", shape = 1, dtype = "int64")
  label_length <- layer_input(name = "label_length", shape = 1, dtype = "int64")

  ctc_func <- function(args){ # args <- list(labels, softmax, input_length, label_length)
    # # the 2 is critical here since the first couple outputs of the RNNs tend to be garbage
    # args[[1]] <-  args[[1]][,3:dim(args[[1]])[2],]
    # #
    k_ctc_batch_cost(y_true = args[[2]], y_pred = args[[1]], input_length = args[[3]], label_length = args[[4]])
  }

  loss <- layer_lambda(list(softmax, labels, input_length, label_length), ctc_func, name = "ctc")

  # create model..
  model <- keras_model(inputs = list(input_layer, labels, input_length, label_length),
                       outputs = loss)
  # summary(model)

  # compile
  ctc_dummy <- function(y_true, y_pred) { y_pred }
  model %>% compile(
    loss = function(y_true, y_pred) ctc_dummy(y_true, y_pred),
    optimizer = optimizer_adam(lr = lr, beta_1 = beta_1, beta_2 = beta_2)
  )
}

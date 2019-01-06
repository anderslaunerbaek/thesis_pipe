
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

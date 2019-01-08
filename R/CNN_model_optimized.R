#' model_CNN
#'
#' A VVG16 model with a conv base which is pretrained on the ImageNet data and the self defined classifier.
#'
#' @param no_classes an integer, number of classes.
#' @param batch_shape a list with `batch_shape`.
#' @param lr the learning rate.
#' @param beta_1 a exponential decay, 1st moment estimates.
#' @param beta_2 a exponential decay, 2nd moment estimates.

#' @importFrom keras application_vgg16 freeze_weights keras_model
#' @importFrom keras layer_dense layer_dropout layer_flatten
#' @importFrom keras compile "%>%" optimizer_adam
#'
#'
#' @return Keras model object
#'
model_CNN <- function(no_classes = 4, batch_shape = list(64, 224,224,3), lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999) {

  vgg16_base <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE,
    input_shape = unlist(batch_shape)[-1]
  )

  model <- vgg16_base$output %>%
    layer_flatten(name = "flatten") %>%
    layer_dropout(rate = 0.5, name = "drop1") %>%
    layer_dense(units = 64, activation = "relu", name = "fc1") %>%
    layer_dropout(rate = 0.5, name = "drop2") %>%
    layer_dense(units = no_classes,
                activation = "softmax",
                name = "predictions")

  model <- keras_model(vgg16_base$input, model)

  freeze_weights(model, to = "flatten")

  # summary(model)

  # compile
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = lr, beta_1 = beta_1, beta_2 = beta_2),
    metrics = c("accuracy")
  )
}

#' model_CNN
#'
#' VVG 16 pretrained on image net
#'
#' @param no_classes an integer, number of classes.
#' @param batch_shape a list with `batch_shape`.
#' @param glo_set a list must be `glo_set`.
#' @importFrom keras application_vgg16
#' @importFrom keras freeze_weights
#' @importFrom keras keras_model_sequential
#' @importFrom keras layer_dense
#' @importFrom keras layer_dropout
#' @importFrom keras layer_flatten
#' @importFrom keras compile
#' @importFrom keras "%>%"
#' @importFrom keras optimizer_adam
#'
#'
#' @return Keras model object
#'
model_CNN <- function(no_classes, batch_shape, glo_set) {

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
    optimizer = optimizer_adam(lr = glo_set$modelling$CNN$lr,
                               beta_1 = glo_set$modelling$CNN$beta1,
                               beta_2 = glo_set$modelling$CNN$beta2),
    metrics = c("accuracy")
  )
}

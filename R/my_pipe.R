#' Title
#'
#' @param pdf_file the path to the VRD pdf file we wish the provide an classification and extract to.
#' @param page_idx default null, provide vector for specific pages in the VRD
#'
#' @return a list with lists
#'
#' @importFrom keras unserialize_model
#' @importFrom dplyr slice left_join mutate bind_rows select n
#' @importFrom tibble as_tibble
#' @importFrom stats predict
#' @importFrom abind abind
#'
#'
#' @export
#'
my_pipe <- function(pdf_file, page_idx = NULL) {

  # get initial values
  image_shape <- get("image_shape")

  # gerenal pre-process
  message("General Pre-processing...")
  pdf <- image_preprocess(pdf_file, page_idx = page_idx)

  # first stage pre-process
  message("First Stage Pre-processing...")
  arr_origin <- first_stage_pre_process(pdf, origin_shape = image_shape$origin)

  # predict origins
  message("Prediction of Origins...")
  # load model
  model <- unserialize_model(readRDS(system.file("extdata", "model_cnn.rds", package = "mypipe")))

  # predict
  preds_proba <- predict(model, arr_origin)
  preds <- as_tibble(preds_proba)
  preds <- mutate(preds,
                  page = 1:n(),
                  class = get("classes")[apply(preds_proba, 1, which.max)],
                  LP = NA,
                  LP_prob = NA,
                  VIN = NA,
                  VIN_prob = NA)
  # rename
  colnames(preds) <- c(get("classes"), "page", "class", "LP", "LP_prob", "VIN", "VIN_prob")

  # apply BR
  DEs <- filter(preds, class == "DEs")
  DEs <- select(DEs, c("page", "class", "LP", "LP_prob", "VIN", "VIN_prob"))

  if(nrow(DEs) > 0) {
    # second stage pre-process
    message("Second Stage Pre-processing...")

    # predict and decode
    DEs <- bind_rows(lapply(1:nrow(DEs), function(i) {
      #
      DEs_sub <- slice(DEs, i)

      arr_text <- second_stage_pre_process(page = pdf[DEs$page[i]],
                                           text_shape = image_shape$extract)


      # load model
      model <- unserialize_model(readRDS(system.file("extdata", "model_cnnrnn.rds", package = "mypipe")))
      images <- abind(arr_text$LP, arr_text$VIN, along = 0)

      # decode
      message("Prediction of LP and VIN...")
      text_decoded <- decode_ctc(model, images)
      LP <- slice(text_decoded, 1)
      VIN <- slice(text_decoded, 2)

      DEs_sub$LP <- LP$decoded_text
      DEs_sub$LP_prob <- LP$prob
      DEs_sub$VIN <- VIN$decoded_text
      DEs_sub$VIN_prob <- VIN$prob

      # return
      DEs_sub
      }))

    # join
    preds <- left_join(select(preds, -c("LP", "LP_prob", "VIN", "VIN_prob")),
                       DEs,
                       by = c("page" = "page", "class" = "class"))
  }
  # return
  as.list(split(preds, seq(nrow(preds))))
}


# utils ----

#' char_to_num
#'
#' @param char a character string
#'
#' @return a vector with numbers which correspond to the character string
#'
char_to_num <- function(char){
  #
  match(char, get("vocab_tot"))
}

#' num_to_char
#'
#' @param num an integer vector
#'
#' @return a character string
#'
num_to_char <- function(num){
  #
  gsub(x = paste0(get("vocab_tot")[num], collapse = ""),
       pattern = "[*]", replacement = "", perl = TRUE)
}

#' Iterative Levenshtein Distance
#'
#' @param s string
#' @param t truth
#'
#' @return an iteger with the number of operations to perform i order to get the correct string
#'
iLD <- function(s, t) {

  if(nchar(t) == 0) {return(nchar(s))}
  rows <- nchar(s) + 1
  cols <- nchar(t) + 1

  dist <- matrix(0, nrow = rows, ncol = cols)
  dist[,1] <- 0:nchar(s)
  dist[1,] <- 0:nchar(t)

  for(ii in 2:cols){
    for(jj in 2:rows){

      if (substr(s, jj-1, jj-1) == substr(t, ii-1, ii-1)) {
        cost <- 0
      } else {
        cost <- 1
      }
      #
      dist[jj, ii] <- min(dist[jj-1, ii] + 1,      # deletion
                          dist[jj, ii-1] + 1,      # insertion
                          dist[jj-1, ii-1] + cost) # substitution
    }
  }
  # return cost
  dist[jj, ii]
}

# General pre-processing stage ----
#' image_preprocess
#'
#' @param pdf_file the path of the the pdf file
#' @param page_idx default null, provide vector for specific pages in the VRD
#'
#' @importFrom magick image_read_pdf image_crop
#' @importFrom magick image_scale image_canny
#' @importFrom magick image_data image_rotate image_median
#' @importFrom tibble as_tibble
#'
#' @return an updated magick pointer
#'
image_preprocess <- function(pdf_file, page_idx = NULL) {
  # load the entire VRD file
  if(!is.null(page_idx)) {
    pdf <- image_read_pdf(pdf_file, pages = page_idx)
  } else {
    pdf <- image_read_pdf(pdf_file)
  }

  # TODO: remove this crop for new data...
  pdf <- image_crop(image = pdf, geometry = "+75+100")

  # process each image
  for (i in 1:length(pdf)){# subset page in pdf
    page <- pdf[i]

    # determine rotation by using a canny filter and the hough transform
    rot <- tryCatch({
      # create the image Z (binary Canny filter image)
      Z <- image_scale(image = page, geometry = "25%")
      Z <- image_canny(image = Z, geometry = "5x1.4+10%+40%")
      Z <- image_data(image = Z, channels = 'gray')
      Z <- aperm(Z, c(3,2,1))
      Z <- array(data = as.integer(Z), dim = dim(Z))

      # determine rotation
      hough_line_votes(Z, degree = seq(-5, 5, by = 0.1), d_rho = 2)
    }, error = function(e) {
      #
      0L
    })

    # deskew the single page
    page <- image_rotate(image = page, degrees = rot)

    # determine the first crop by determine the OTSUs theshold
    page_OTSU <- image_median(image = page, radius = 5)
    page_OTSU <- image_data(page_OTSU, 'gray')
    page_OTSU <- aperm(page_OTSU, c(3,2,1))
    page_OTSU <- array(data = as.integer(page_OTSU), dim = dim(page_OTSU))

    # the OTSUs threshold
    L <- OTSU(as.integer(page_OTSU))

    # determine the smalles rectangular crop for rotated page
    idx <- as_tibble(which(page_OTSU < L, arr.ind = TRUE))
    rows <- c(min(idx$dim1), max(idx$dim1), max(idx$dim1) - min(idx$dim1))
    cols <- c(min(idx$dim2), max(idx$dim2), max(idx$dim2) - min(idx$dim2))

    if (all(is.finite(c(cols, rows)))) {
      str_crop <- paste0(cols[3],"x",rows[3],"+",cols[1],"+",rows[1])
    } else {
      str_crop <- paste0(dim(page_OTSU)[2],"x",dim(page_OTSU)[1],"+",0,"+",0)
    }

    # update the magick pointer for page "i"
    pdf[i] <- image_crop(image = page, geometry = str_crop)
  }
  # return magick pointer
  pdf
}

#' OTSU
#'
#' The function determine the Otsus threshold
#'
#' @param input a vector of gray-scale pixel values
#'
#' @return The OTSUs theshold
#'
OTSU <- function(input) {
  # inilitize
  counts <- table(input)
  num_bins <- length(counts)

  # variables names are chosen to be similar to the formulas in the Otsu paper.
  p <- counts / sum(counts)
  omega <- cumsum(p)

  #
  mu <- cumsum(p * as.integer(names(counts)))
  mu_t <- mu[length(mu)]

  #
  sigma_b_squared <- (mu_t * omega - mu)^2 / (omega * (1 - omega))
  sigma_b_squared <- ifelse(is.nan(sigma_b_squared),0,sigma_b_squared)

  # find the location of the maximum value of sigma_b_squared.
  idx <- mean(which(sigma_b_squared == max(sigma_b_squared)))

  # normalize the threshold to the range [0, 255] and return
  as.integer((idx - 1) / (num_bins - 1) * 255)
}

#' hough_line_votes
#'
#' @param Z a binary Canny filter image
#' @param degree vector with possible rotations
#' @param d_rho the resolution of rho
#'
#' @importFrom tibble as_tibble
#'
#' @return an estimated rotation in degrees
#'
hough_line_votes <- function(Z, degree, d_rho) {
  # initial values
  rows <- dim(Z)[1]
  cols <- dim(Z)[2]
  R <- 1/2 * sqrt(cols^2 + rows^2)
  rho_k <- seq(-R, R, by = d_rho)
  M <- length(degree)
  N <- length(rho_k)
  H <- matrix(0, nrow = M, ncol = N)
  idx <- as_tibble(which(Z > 0, arr.ind = TRUE))

  # Hough transform
  for (i in 1:nrow(idx)) {
    pk <- idx$dim1[i] * cos(degree) + idx$dim2[i] * sin(degree)
    #
    for (l in 1:N) {
      k <- which.min(abs(rho_k[l]- pk))
      H[k, l] <- H[k, l] + 1
    }
  }

  # find rotation
  idx_theta <- which(H == max(H), arr.ind = TRUE)
  # return rotation
  degree[idx_theta[1,1]]
}

#' crop_func
#'
#' @param page magick pointer
#' @param crop_area a string with the crop geometry
#' @param text_shape shape of the output array
#' @param median_filter_size vector with radius for two median filters
#' @param pad the size of the padding around the crop area
#'
#' @importFrom magick image_crop image_data image_read image_border
#' @importFrom magick image_median image_resize image_info
#' @importFrom tibble as_tibble
#' @importFrom dplyr select filter arrange summarise
#' @importFrom grDevices x11
#'
#' @return an array with the text crop
#'
crop_func <- function(page, crop_area, text_shape, median_filter_size, pad) {
  # initial values
  img <- image_crop(image = page, geometry = crop_area)
  img <- image_data(img, "gray")
  img <- aperm(img, c(3,2,1))
  img <- array(data = as.integer(img), dim = dim(img))

  # OTSU
  L <- OTSU(as.integer(img))

  img[which(img < L, arr.ind = TRUE)] <- 0
  img[which(img >= L, arr.ind = TRUE)] <- 255

  if (median_filter_size[1] == 0) {
    img_resize <- image_read(img / 255)
  } else {
    # Determine crop
    img_crop <- image_read(img / 255)
    img_crop <- image_median(image = img_crop, median_filter_size[1])
    img_crop <- image_data(image = img_crop, "gray")
    img_crop <- aperm(img_crop, c(3,2,1))
    img_crop <- array(data = as.integer(img_crop), dim = dim(img_crop))

    idx <- as_tibble(which(img_crop == 0, arr.ind = TRUE))
    idx <- select(idx, -c("dim3"))
    idx <- arrange(idx, dim1, dim2)
    idx <- summarise(idx,
                     x11 = max(min(dim1) - pad, 0),
                     x21 = max(dim1),
                     x12 = max(min(dim2) - pad, 0),
                     x22 = max(dim2),
                     x1d = x21 - x11 + pad * 2,
                     x2d = x22 - x12 + pad * 2)
    img_resize <- image_read(img / 255)
    img_resize <- image_crop(image = img_resize,
                             geometry = paste0(idx$x2d,"x",idx$x1d,"+",idx$x12,"+",idx$x11))



  }

  img_pad <- image_median(image = img_resize, radius = median_filter_size[2])
  img_pad <- image_resize(image = img_pad,
                          geometry = geometry_size_pixels(width = text_shape[1],
                                                          height = text_shape[2],
                                                          preserve_aspect = TRUE))

  width <- image_info(img_pad)$width
  height <- image_info(img_pad)$height
  pad_str <- paste0(floor((text_shape[1] - width) / 2), "x", floor((text_shape[2] - height) / 2))

  img_out <- image_border(image = img_pad, color = "white", geometry = pad_str)
  img_out <- image_data(img_out, "gray")
  img_out <- aperm(img_out, c(2,3,1))
  img_out <- array(data = as.integer(img_out), dim = dim(img_out))


  # update if odd dimensions
  arr <- array(255, dim = text_shape)
  arr[1:dim(img_out)[1],1:dim(img_out)[2],] <- img_out

  # return
  arr
}



# BR stage ----

#' format_text_output
#'
#' This function does apply the BR for the VIN text
#'
#' @param text_pred a character string
#'
#' @return a cleaned character string
#'
format_text_output <- function(text_pred) {

  text_pred <- gsub(x = toupper(text_pred),
                    pattern = paste0("[^", get("vocab_clean"), "]"),
                    replacement = "", perl = TRUE)

  if (nchar(text_pred) == 17) {
    # BR
    text_pred <- gsub(x = text_pred, pattern = "I",  replacement = "1", perl = TRUE)
    text_pred <- gsub(x = text_pred, pattern = "Q|O",  replacement = "0", perl = TRUE)
  }

  # return
  text_pred
}



# First stage ----
#' first_stage_pre_process
#'
#' @param pdf magick pointer
#' @param origin_shape the shape of the array
#'
#' @importFrom magick image_resize geometry_size_pixels image_data
#'
#' @return an array with resized pages
#'
first_stage_pre_process <- function(pdf, origin_shape) {

  # inital values
  n <- length(pdf)
  arr_origin <- array(data = NA, dim = c(n, origin_shape))

  # from magick pointer to array
  for (i in 1:n){
    x <- image_resize(image = pdf[i],
                      geometry = geometry_size_pixels(width = origin_shape[1],
                                                      height = origin_shape[2],
                                                      preserve_aspect = FALSE))
    x <- image_data(x, 'rgb')
    x <- aperm(x, c(3,2,1))

    arr_origin[i,,,] <- array(as.integer(x), dim = dim(x))
  }

  # return
  arr_origin
}
# Second stage ----
#' Title
#'
#' @param page magick pointer
#' @param text_shape output shape for text field
#' @param median_filter_size a vector of two, default c(5,3)
#' @param pad the padding, default 4
#'
#' @return a list with two array, one for LP and one for VIN
#'
second_stage_pre_process <- function(page, text_shape, median_filter_size = c(5,3), pad = 4) {

  # produce image for LP and VIN
  width <- image_info(page)$width
  height <- image_info(page)$height

  # get intial value
  ca <- get("ca")
  LP_cal <-  round(ca$LP_pct * c(height, width, height, width))
  VIN_cal <-  round(ca$VIN_pct * c(height, width, height, width))

  # lisence plate
  img_LP <- tryCatch({
    crop_func(page, crop_area = paste0(LP_cal[1],"x",LP_cal[2],"+",LP_cal[3],"+",LP_cal[4]),
              text_shape = text_shape,
              median_filter_size = median_filter_size,
              pad = pad)
  }, error = function(e) {
    array(0, dim = text_shape)
  })

  # vehicle identification number
  img_VIN <- tryCatch({
    crop_func(page,
              crop_area = paste0(VIN_cal[1],"x",VIN_cal[2],"+",VIN_cal[3],"+",VIN_cal[4]),
              text_shape = text_shape,
              median_filter_size = median_filter_size,
              pad = pad)
  }, error = function(e) {
    array(0, dim = text_shape)
  })

  # return
  list("LP" = img_LP,
       "VIN" = img_VIN)
}

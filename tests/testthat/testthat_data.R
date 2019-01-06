library(mypipe)

test_that("Check for nesseary global variables", {

  expect_match(class(ca), "list")

  expect_match(class(classes), "character")

  expect_match(class(image_shape), "list")
  expect_match(class(image_shape$origin), "integer")
  expect_match(class(image_shape$extract), "integer")

  expect_match(class(vocab_clean), "character")
  expect_match(class(vocab_tot), "character")

  #


})


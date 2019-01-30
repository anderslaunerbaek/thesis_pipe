
<!-- README.md is generated from README.Rmd. Please edit that file -->
The Technical Pipeline
======================

[![Build Status](https://travis-ci.com/anderslaunerbaek/thesis_pipe.svg?branch=master)](https://travis-ci.com/anderslaunerbaek/thesis_pipe)

<!--
## Performance Tables





Partition        Kappa....   Error.Rate.... 
---------------  ----------  ---------------
Train Set        99.7117     0.1651         
Validation Set   98.5066     0.8557         
Test Set         98.4293     0.8998         




Partition        LER      Error.Rate.... 
---------------  -------  ---------------
Train Set        1.684    56.5776        
Validation Set   2.5885   74.2798        
Test Set         2.6472   72.8618        



First.Stage   Second.Stage   N       Percentage.... 
------------  -------------  ------  ---------------
Correct       Correct        35      1              
Correct       Wrong          573     18             
Correct       -              2,575   81             
Wrong         -              36      1              

-->
Todos
-----

-   Unit tests
-   Grad-CAM CRNN model. Per time step, per collapsed..
-   Add code coverage (create codecov.yml, activate codecov.io and get batch to readme...)

Link to Report
--------------

Install Package
---------------

``` r
# install.packages("devtools")
devtools::install_github("anderslaunerbaek/thesis_pipe")
```

Run Example
-----------

``` r
pdf_file <- "./dev/data/my_VRD_pdf_file.pdf"

# Load library
library(thesispipe)

# Perform classification and text predictions
my_pipe(pdf_file)
```

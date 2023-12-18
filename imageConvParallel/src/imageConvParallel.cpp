#include <RcppArmadillo.h>
#include <RcppParallel.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]

using namespace arma;
using namespace RcppParallel;

struct ImageConvolution : public Worker {
  const RcppParallel::RMatrix<double> img;
  const RcppParallel::RMatrix<double> kernel;
  RcppParallel::RMatrix<double> result;
  
  ImageConvolution(const Rcpp::NumericMatrix& img, const Rcpp::NumericMatrix& kernel, Rcpp::NumericMatrix result)
    : img(img), kernel(kernel), result(result) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    int imgRows = img.nrow();
    int imgCols = img.ncol();
    
    for (std::size_t i = begin; i < end; i++) {
      int row = i % imgRows;
      int col = i / imgRows;
      double rst = 0.0;
      
      for (int k = -1; k < 2; k++) {
        for (int l = -1; l < 2; l++) {
          int resultI = row + k;
          int resultJ = col + l;
          resultI = mirrorIndex(resultI, imgRows);
          resultJ = mirrorIndex(resultJ, imgCols);
          rst += img(resultI, resultJ) * kernel(k + 1, l + 1);
        }
      }
      
      if (rst < 0) rst = 0;
      if (rst > 1) rst = 1;
      
      result(row, col) = rst;
    }
  }
  
  int mirrorIndex(int fetchI, int length) {
    if (fetchI < 0)
      fetchI = -fetchI - 1;
    if (fetchI >= length)
      fetchI = 2 * length - fetchI - 1;
    return fetchI;
  }
};

// [[Rcpp::export]]
Rcpp::NumericMatrix imageConvParallel(Rcpp::NumericMatrix img, Rcpp::NumericMatrix kernel) {
  int imgRows = img.nrow();
  int imgCols = img.ncol();
  Rcpp::NumericMatrix result(imgRows, imgCols);
  
  ImageConvolution imageConvolution(img, kernel, result);
  parallelFor(0, imgRows * imgCols, imageConvolution);
  
  return result;
}
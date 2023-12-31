// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// imageConvParallel
Rcpp::NumericMatrix imageConvParallel(Rcpp::NumericMatrix img, Rcpp::NumericMatrix kernel);
RcppExport SEXP _imageConvParallel_imageConvParallel(SEXP imgSEXP, SEXP kernelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type img(imgSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type kernel(kernelSEXP);
    rcpp_result_gen = Rcpp::wrap(imageConvParallel(img, kernel));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_imageConvParallel_imageConvParallel", (DL_FUNC) &_imageConvParallel_imageConvParallel, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_imageConvParallel(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

#' Housing dataset
#'
#' @description Kurzer Datensatz-Text (Worum geht’s?).
#' @format Ein data.frame mit … Zeilen und … Variablen:
#' \describe{
#'   \item{var1}{Beschreibung}
#'   \item{var2}{Beschreibung}
#' }
#' @source Quelle / DOI / URL
#' @usage data(insurance)
#' @examples
#' data(insurance)
#' str(insurance)
"insurance"

## code to prepare `insurance` dataset goes here
insurance <- read.csv("data-raw/insurance.csv", na.strings = c("", "NA"))

usethis::use_data(insurance, overwrite = TRUE)

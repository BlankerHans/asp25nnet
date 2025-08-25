#' Housing dataset
#'
#' @description Kurzer Datensatz-Text (Worum geht’s?).
#' @format Ein data.frame mit … Zeilen und … Variablen:
#' \describe{
#'   \item{var1}{Beschreibung}
#'   \item{var2}{Beschreibung}
#' }
#' @source Quelle / DOI / URL
#' @usage data(housing)
#' @examples
#' data(ca_housing)
#' str(ca_housing)
"ca_housing"

## code to prepare `ca_housing` dataset goes here
ca_housing <- read.csv("data-raw/ca_housing.csv", na.strings = c("", "NA"))


usethis::use_data(ca_housing, overwrite = TRUE)


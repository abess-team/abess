+rm(list = ls())
gc(reset = TRUE)
library(data.table)
library(fastDummies)
working_path <- getwd()
data_raw_path <- paste0(working_path, "/data-raw")
setwd(data_raw_path)

#############################################################
####################### Crime Dataset #######################
#############################################################
crime <- fread("crime.txt", data.table = FALSE, header = FALSE)
col_name <- fread("crime_colnames.txt",
  data.table = FALSE,
  header = FALSE
)[, 1]
y_col_name <- fread("crime_Y_colnames.txt",
  data.table = FALSE,
  header = FALSE
)[, 1]
colnames(crime) <- col_name
crime[crime == "?"] <- NA

# row missing analysis
sum(apply(crime, 1, anyNA))
summary(apply(crime, 1, function(x) {
  mean(is.na(x))
}))

# column missing analysis
sum(apply(crime, 2, anyNA))
missing_prop <- apply(crime, 2, function(x) {
  mean(is.na(x))
})
sum(missing_prop > 0.5)
missing_prop[missing_prop > 0.5]

# remove column with large missing proportion
delete_col <- which(missing_prop > 0.5)
crime <- crime[, -delete_col]

# remove row with missing
sum(apply(crime, 1, anyNA))
crime <- crime[complete.cases(crime), ]

##########################################
########### col-wise analysis ############
##########################################

# response data
y_index <- which(colnames(crime) %in% y_col_name)
y <- crime[, y_index]
y <- apply(y, 2, as.numeric)
y <- as.data.frame(y)

# exclude response data:
crime <- crime[, -y_index]

# communityname: community name
length(unique(crime[["communityname"]])) # too sparse to be a dummy variable
crime[["communityname"]] <- NULL

# fold: the number for non-random 10 fold cross validation
crime[["fold"]] <- NULL

# state: US state
length(unique(crime[["State"]]))
table(crime[["State"]]) # remove some category

to_remove_state <- names(table(crime[["State"]]))[as.vector(table(crime[["State"]])) < 20]
crime[["State"]][crime[["State"]] %in% to_remove_state] <- "other"
table(crime[["State"]])
state_data <- crime[, "State", drop = FALSE]
state_data <- dummy_cols(
  .data = state_data, select_columns = "State",
  remove_first_dummy = TRUE, remove_selected_columns = TRUE
)

# feature interaction:
crime[["State"]] <- NULL
crime <- apply(crime, 2, as.numeric)
crime <- as.data.frame(crime)
crime_interaction <- model.matrix(~ .^2, data = crime)
dim(crime_interaction)
head(crime_interaction[, 1:6])
crime_interaction <- crime_interaction[, -1] # remove intercept term
head(crime_interaction[, 1:6])

# X data:
x <- crime_interaction
y <- y[["violentPerPop"]]

# save data:
crime <- cbind.data.frame(y, x)

# sample:
set.seed(1)
part_index <- 1:nrow(x)
part_index <- sample(part_index,
  size = 500,
  replace = FALSE
)
crime <- crime[part_index, ]

save(crime, file = "crime.rda")

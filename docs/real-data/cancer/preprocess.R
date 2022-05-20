load("chin.RData")
x <- chin[["x"]]
head(x[, 1:6])
write.csv(x, file = "chin_x.txt", row.names = FALSE, col.names = FALSE)
y <- chin[["y"]]
y <- as.matrix(as.numeric(y) - 1)
write.csv(y, file = "chin_y.txt", row.names = FALSE, col.names = FALSE)

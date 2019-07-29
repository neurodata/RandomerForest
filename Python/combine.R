
require(data.table)

f <- grep("*tsv", dir("RESULTS", full.names = TRUE), value = TRUE)

dat <- vector("list", length(f))



for(i in 1:length(f)){
	tryCatch({
		dat[[i]] <- read.table(f[i], sep = "\t", header = TRUE)
	},
	error=function(error_message){
		message("Something happend")
		message(error_message)
		return(NA)
	})
}

#dat <- lapply(f, function(x) read.table(x, sep = "\t", header = TRUE))

out <- Reduce(rbind, dat)
out$p <- Im(as.complex(out$mtry))
out$mtry <- Re(as.complex(out$mtry))

out <- as.data.table(out)


write.table(out, file = "AllData.tsv", sep = "\t", row.names = FALSE)

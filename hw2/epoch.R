# MyData <- read.csv(file="/home/luke/Downloads/logreg_epoch_test.csv", header=TRUE, sep=",")
# NewData <- MyData
# 
# for(i in length(MyData[,2])){
#  NewData[i,2] = MyData[i,2] + rnorm(1,0,0.01) 
# }
# plot(NewData,main = "Accuracy vs. Epochs", xlab = "Epoch", ylab="Accuracy",type="p",ylim=c(0.5,1))

accuracy1 = read.csv(file="/keybase/private/lukemeszar/CSCI_5622/csci_5622_hws-master/hw2/accuracy1.csv", header=TRUE, sep=",")
accuracy01 = read.csv(file="/keybase/private/lukemeszar/CSCI_5622/csci_5622_hws-master/hw2/accuracy0-1.csv", header=TRUE, sep=",")
accuracy001 = read.csv(file="/keybase/private/lukemeszar/CSCI_5622/csci_5622_hws-master/hw2/accuracy0-01.csv", header=TRUE, sep=",")
accuracy0001 = read.csv(file="/keybase/private/lukemeszar/CSCI_5622/csci_5622_hws-master/hw2/accuracy0-001.csv", header=TRUE, sep=",")
plot(accuracy1,type="l",col="red", main="Effect of Eta on Accuracy", xlab = "Number of training samples", ylab='Accuracy')
lines(accuracy01,col="green")
lines(accuracy001,col="blue")
lines(accuracy0001,col="orange")
legend(x = "bottomright", legend=c("Eta = 1", "Eta = 0.1", "Eta = 0.01", "Eta = 0.001"),
       col=c("red", "green", "blue","orange"), lty=1:2, cex=0.8)
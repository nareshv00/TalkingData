library(data.table)
library(FeatureHashing)
library(Matrix)
library(xgboost)
library(dplyr)
library(slam)
# Reading and merging the data  
ReadData  = function(x) fread(x,colClasses = "character",integer64=getOption("datatable.integer64"))
#function to convert data to string
toStr  = function(x) paste(x, collapse = ",")
#function to calculate number off activations and deactivations of events, binary sums
binarySum= function(x) sum(as.integer(x))

#Reading the data
app_events = ReadData("../Data/app_events.csv/app_events.csv")
#Reading the app label data
app_labels= ReadData("../Data/app_labels.csv/app_labels.csv")
#reading the label_categories data
label_category= ReadData("../Data/label_categories.csv/label_categories.csv")
#merging label categories with label id
app_labels <- merge(label_category, app_labels, by = "label_id", all.x = T)
#merging app lables with app data
app_labels <- app_labels[,.(labelCategory=toStr(category)),by=app_id]
#merging app_labels with app_events
app_events <- merge(app_events, app_labels, by = "app_id", all.x = T)

#aggregating the app_events with number of installation and activations of application
app_events = app_events[ , .(apps = toStr(app_id),appCategory=toStr(labelCategory),isInstalled=binarySum(is_installed),isActive=binarySum(is_active)), by = event_id]

#looking at the application events data using glimpse function
glimpse(app_events)

events <- ReadData("../Data/events.csv/events.csv")
events <- merge(events, app_events, by = "event_id", all.x = T)
#taking hour from the time stamp to do analysis
events$eventTime=as.integer(format(as.POSIXct(events$timestamp, format="%Y-%m-%d %H:%M"), format="%H"))
#taking month from the time stamp to do analysis
events$eventDay=as.integer(format(as.POSIXct(events$timestamp, format="%Y-%m-%d %H:%M"), format="%d"))

#taking both hour directly without summing it 
#events$eventTime=format(as.POSIXct(events$timestamp, format="%Y-%m-%d %H:%M"), format="%H")

#I am not sure what I am looking for , But I am going to sum EventsTimes for the specific device,
#to check if there is any trend, I will take average of this
events <- events[ , .(apps = toStr(apps),appCategory=toStr(appCategory),
isInstalled=sum(isInstalled,na.rm=T),isActive=sum(isActive,na.rm = T)
,deviceTime=mode(eventTime),deviceDay=round(mean(eventDay),0)), by = device_id]
rm(app_events)


# Merge bag-of-apps and brand data into train and test users 
users_train <- ReadData("../Data/gender_age_train.csv/gender_age_train.csv")
users_test  <- ReadData("../Data/gender_age_test.csv/gender_age_test.csv")
brands      <- ReadData("../Data/phone_brand_device_model.csv/phone_brand_device_model.csv")
#removing duplicates from phone brands
brands      <- brands[!duplicated(brands$device_id), ]

MergeTalk <- function(x, y) merge(x, y, by = "device_id", all.x = T)
users_train <- MergeTalk(users_train, events)
users_train <- MergeTalk(users_train, brands)
users_test  <- MergeTalk(users_test, events)
users_test  <- MergeTalk(users_test, brands)

#feature engineering isInstalled and isActive
#Handling NA's , and I am assuming users with NA's in isInstalled and isActive as the
#users who are not using any applications or services or calling them as Not available.
#I will make this column as a factor
#isActive
#NA=Not available
#users_train$isActive<100 as starters
#users_train$isActive>100 & users_train$isActive<500 as moderate
#users_train$isActive>500 & users_train$isActive<5000 as High
#users_train$isActive>5000 as very high

isActiveCategory=function(users)ifelse(is.na(users$isActive)==T,"Not available",
           ifelse(users$isActive<=100,"Starters",
          ifelse(users$isActive>100 & users$isActive<=500,"Moderate",
          ifelse(users$isActive>500 & users$isActive<=5000,"High","Very High"
                               ))))
# isInstalled
#as the number of applications installed on a cellphone varies vastly , I am assuming a 
#different condition for the isInstalled variable

isInstalledCategory=function(users)ifelse(is.na(users$isInstalled)==T,"Not available",
                                       ifelse(users$isInstalled<=50,"Starters Applications",
                                              ifelse(users$isInstalled>50 & users$isInstalled<=100,"Moderate Applications",
                                                     ifelse(users$isInstalled>100 & users$isInstalled<=500,"High Applications","Very High Applications"
                                                     ))))
# isDayTime
#as the number of applications installed on a cellphone varies vastly , I am assuming a 
#different condition for the isInstalled variable
#checking the histogram to feature it correct
hist(users_train$deviceTime)
#feature engineering deviceTime to isDayTime
isDayTime=function(users)ifelse(is.na(users$deviceTime)==T,"Not available",
                                          ifelse(users$deviceTime>=2 & users$deviceTime<=8,"Early Morning",
                                                 ifelse(users$deviceTime>8 & users$deviceTime<=12,"Morning",
                                                        ifelse(users$deviceTime>12 & users$deviceTime<=20,"Evening"
                                                               ,ifelse(users$deviceTime>20 & users$deviceTime<=24,"Night","Mid Night"
                                                        )))))

#calling isActiveCategory function on users_train
users_train$isActiveCategory=as.factor(isActiveCategory(users_train))
#calling isActiveCategory on users_test
users_test$isActiveCategory=as.factor(isActiveCategory(users_test))

#calling isInstalledCategory function on users_train
users_train$isInstalledCategory=as.factor(isInstalledCategory(users_train))
#calling isInstalledCategory on users_test
users_test$isInstalledCategory=as.factor(isInstalledCategory(users_test))

#calling isDayTime function on users_train
users_train$isDayTime=as.factor(isDayTime(users_train))
#calling isDayTime on users_test
users_test$isDayTime=as.factor(isDayTime(users_test))

#changing this

#looking at the merged users train data using glimpse function
#cleaning data
glimpse(users_train)

# FeatureHash brand and app data to sparse matrix
b <- 2 ^ 15
f <- ~deviceDay+isDayTime+isInstalledCategory+isActiveCategory+ phone_brand + device_model + split(apps, delim = ",")+split(appCategory,delim=",")- 1
X_train <- hashed.model.matrix(f, users_train, b)
X_test  <- hashed.model.matrix(f, users_test,  b)

#removing columns with zero variance
# Validate xgboost model
Y_key <- sort(unique(users_train$group))
Y     <- match(users_train$group, Y_key) - 1

model <- sample(1:length(Y), 50000)
valid <- (1:length(Y))[-model]

param <- list(objective = "multi:softprob", num_class = 12,
              booster = "gblinear", eta = 0.01,
              eval_metric = "mlogloss",depth=8,lambda=0.5,lambda_bias=0.5,
              alpha=0.5,
              colsample_bytree=0.7,
              num_parallel_tree=1)

dmodel <- xgb.DMatrix(X_train[model,], label = Y[model])
dvalid <- xgb.DMatrix(X_train[valid,], label = Y[valid])
watch  <- list(model = dmodel, valid = dvalid)

m1 <- xgb.train(data = dmodel, param, nrounds = 130,
                watchlist = watch)


# Use all train data and predict test
dtrain <- xgb.DMatrix(X_train, label = Y)
dtest  <- xgb.DMatrix(X_test)

m2 <- xgb.train(data = dtrain, param, nrounds = 150)

out <- matrix(predict(m2, dtest), ncol = 12, byrow = T)
out <- data.frame(device_id = users_test$device_id, out)
names(out)[2:13] <- Y_key
write.csv(out, file = "sub1.csv", row.names = F)


#working with ggplots to check the patterns in the data

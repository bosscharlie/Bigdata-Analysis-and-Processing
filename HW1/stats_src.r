library('readxl')
df=read_excel('./data.xlsx')
out_pdf <- function(fname,x){
    png(paste("epdf_",fname,".png",sep=""))
    epdfPlot(x,discrete=FALSE)
    dev.off()
}

epdf_and_normtest <- function() {
    print("Q3.a")
    if(!require('EnvStats')){
        install.packages('EnvStats')
        library('EnvStats')
    }
    if(!require('nortest')) {
        install.packages('nortest')
        library('nortest')
    }
    x=df$平均年龄

    # plot epdf
    out_pdf("all",x)

    #normality test
    res = lillie.test(x)
    print(res)
}

category_normtest <- function(){
    print("Q3.b")
    age_1=subset(df,群类别==1)$平均年龄
    out_pdf("1",age_1)
    print(lillie.test(age_1))
    age_2=subset(df,群类别==2)$平均年龄
    out_pdf("2",age_2)
    print(lillie.test(age_2))
    age_3=subset(df,群类别==3)$平均年龄
    out_pdf("3",age_3)
    print(lillie.test(age_3))
    age_4=subset(df,群类别==4)$平均年龄
    out_pdf("4",age_4)
    print(lillie.test(age_4))
    age_5=subset(df,群类别==5)$平均年龄
    out_pdf("5",age_5)
    print(lillie.test(age_5))
    # test the homogeneity of variances.
    fligner.test(平均年龄~群类别,data=df)
}

anova_test <- function(){
    print("Q3.c")
    df$群类别=ordered(df$群类别,levels=c(1,2,3,4,5))
    if(!require('ggpubr')) {
        install.packages('ggpubr')
        library('ggpubr')
    }
    if(!require('stats')) {
        install.packages('stats')
        library('stats')
    }
    png("box.png")
    ggboxplot(df,x="群类别",y="平均年龄",ylab="avg_age",xlab='category')
    dev.off()
    print(summary(aov(平均年龄~群类别,data=df)))
    kruskal.test(平均年龄~群类别,data=df)
}

regression <- function(){
    print("Q4.a")
    regression_data=subset(df,会话数>=20)
    png("lm_nores.png")
    plot(无回应比例~平均年龄,data=regression_data,ylab="noresponse_rate",xlab="avg_age")
    lmres=lm(无回应比例~平均年龄,data=regression_data)
    abline(lmres,col="red")
    dev.off()
    print(summary(lmres))

    png("lm_night.png")
    plot(夜聊比例~平均年龄,data=regression_data,ylab="night_rate",xlab="avg_age")
    lmres=lm(夜聊比例~平均年龄,data=regression_data)
    abline(lmres,col="red")
    dev.off()
    print(summary(lmres))

    png("lm_pic.png")
    plot(图片比例~平均年龄,data=regression_data,ylab="pic_rate",xlab="avg_age")
    lmres=lm(图片比例~平均年龄,data=regression_data)
    abline(lmres,col="red")
    dev.off()
    print(summary(lmres))
}

weighted_regression <- function(){
    print("Q4.b")
    lmres=lm(无回应比例~群人数+消息数+稠密度+性别比+平均年龄+年龄差+地域集中度+手机比例,data=df,weights=会话数)
    print(summary(lmres))
    lmres=lm(夜聊比例~群人数+消息数+稠密度+性别比+平均年龄+年龄差+地域集中度+手机比例,data=df,weights=会话数)
    print(summary(lmres))
    lmres=lm(图片比例~群人数+消息数+稠密度+性别比+平均年龄+年龄差+地域集中度+手机比例,data=df,weights=会话数)
    print(summary(lmres))
}

logistical_regression <- function(){
    print("Q4.c")
    if(!require('tidymodels')) {
        install.packages('tidymodels')
        library('tidymodels')
    }
    lgrdf=subset(df,群类别==1|群类别==4)
    lgrdf$群类别=as.character(lgrdf$群类别)
    lgrdf$群类别[lgrdf$群类别 == "1"] <- 'class1'
    lgrdf$群类别[lgrdf$群类别 == "4"] <- 'class4'
    lgrdf$群类别=as.factor(lgrdf$群类别)
    data_set=initial_split(lgrdf,prop=0.8,strata = 群类别)
    train_set=training(data_set)
    test_set=testing(data_set)
    model = logistic_reg(mixture=double(1),penalty=double(1)) %>%
        set_engine("glm") %>%
        set_mode("classification") %>%
        fit(群类别~群人数+消息数+稠密度+性别比+平均年龄+年龄差+地域集中度+手机比例+会话数+无回应比例+夜聊比例+图片比例,data=train_set)
    print(tidy(model))
    pred_class=predict(model,new_data=test_set,type="class")
    pred_proba=predict(model,new_data=test_set,type="prob")
    results=test_set %>%
        select(群类别) %>%
        bind_cols(pred_class,pred_proba)
    accuracy(results,truth = 群类别,estimate=.pred_class)
}

sample_aov <- function(){
    print("Q5")
    # Simple Random Sampling
    print("Simple Random Sampling")
    fs=vector(mode="numeric",length=10)
    for(i in 1:10){
        sample_index=sample(1:2040,200)
        di=df[sample_index,]
        res=summary(aov(平均年龄~群类别,data=di))
        fs[i]=res[[1]][["F value"]][1]
    }
    print(fs)
    print("mean:")
    print(mean(fs))
    print("std dev:")
    print(sd(fs))

    # Stratified Random Sampling
    print("Stratified Random Sampling")
    if(!require('sampling')) {
        install.packages('sampling')
        library('sampling')
    }
    fss=vector(mode="numeric",length=10)
    for(i in 1:10){
        ssindex=strata(df,c("群类别"),size=c(484/10,300/10,196/10,425/10,635/10),method="srswor")
        ssi=df[ssindex$ID_unit,]
        res=summary(aov(平均年龄~群类别,data=ssi))
        fss[i]=res[[1]][["F value"]][1]
    }
    print(fss)
    print("mean:")
    print(mean(fss))
    print("std dev:")
    sd(fss)
}

epdf_and_normtest()
category_normtest()
anova_test()
regression()
weighted_regression()
logistical_regression()
sample_aov()
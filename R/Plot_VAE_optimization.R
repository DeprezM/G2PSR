## Author : Marie Deprez
## Title : Graph VAE optimization metrics

## Upload needed packages
library(ggplot2)

setwd("/user/mdeprez/home/Documents/Data_ADNI/pathways-ae-master")


# Quick check of the metrics shape --------------------------------------------------------------------------
#### Alpha parameter - sparsity

ctr_values <- read.table("/user/mdeprez/home/Documents/Data_ADNI/Plink_LD/p_values_100G_80k.csv",
                         header = T, sep = ",")

ctr_values$X <- seq(from = 0, to = 80000, by = 100)
df <-  melt(ctr_values ,  id.vars = 'X', variable.name = 'series')

gg <- ggplot(df, aes(x = X, y = value, color = series), show.legend = F)+
  geom_line(size = 0.25, alpha = 0.9, show.legend = F) +
  scale_color_grey()+
  theme_classic() +
  xlab("Number of iteration (epoch)") +
  ylab(expression(paste("Alpha probability p(", alpha,")")))+
  geom_line(aes(y = 0.05), color = "black", size = 0.6, linetype = "dashed") +
  guides(col = F)+
  geom_line(data = ctr_values[, c("X", "APOE")], aes(x = X, y = APOE), color = "red", size = 1.2, show.legend = T) +
  theme(legend.position = "none",
        text = element_text(size = 15)) 
gg

pdf("alpha.pdf", width=6, height=4, useDingbats=FALSE)
gg
dev.off()


#### Mu parameter - weight 
ctr_values <- read.table("/user/mdeprez/home/Documents/Data_ADNI/Plink_LD/mu_values_100G.csv",
                         header = T, sep = ",")

ctr_values$X <- seq(from = 0, to = 60000, by = 100)
df <-  melt(ctr_values ,  id.vars = 'X', variable.name = 'series')

gg <- ggplot(df, aes(x = X, y = value, color = series), show.legend = F)+
  geom_line(size = 0.25, alpha = 0.9, show.legend = F) +
  theme_classic() +
  xlab("Number of iteration (epoch)") +
  ylab(expression(paste("Weight")))+
  guides(col = F)+
  geom_line(data = ctr_values[, c("X", "APOE")], aes(x = X, y = APOE), color = "red4", size = 1.2, show.legend = T) +
  theme(legend.position = "none",
        text = element_text(size = 15)) 
gg

pdf("weight.pdf", width=6, height=4, useDingbats=FALSE)
gg
dev.off()

# Evaluate VAE optimazation --------------------------------------------------------------------------

nb_epoch <- 50

ctr_values <- read.table(paste0("/user/mdeprez/home/Documents/Data_ADNI/Plink_LD/p_values_100G_", 
                                nb_epoch, "k.csv"),
                         header = T, sep = ",")


ctr_values$X <- seq(from = 0, to = nb_epoch*1000, by = 100)
df <-  melt(ctr_values ,  id.vars = 'X', variable.name = 'series')

gg <- ggplot(df, aes(x = X, y = value, color = series), show.legend = F)+
  geom_line(size = 0.25, alpha = 0.9, show.legend = F) +
  scale_color_grey()+
  theme_classic() +
  xlab("Number of iteration (epoch)") +
  ylab(expression(paste("Alpha probability p(", alpha,")")))+
  geom_line(aes(y = 0.05), color = "black", size = 0.6, linetype = "dashed") +
  guides(col = F)+
  geom_line(data = df[df$series %in%top_genes, ], aes(x = X, y = value, color = series), size = 1.2, show.legend = T) +
  theme(legend.position = "none",
        text = element_text(size = 15)) 
gg




top_genes <- colnames(ctr_values)[ctr_values[601,] < 0.05]
top_genes_70 <- colnames(ctr_values)[ctr_values[701,] < 0.05]
top_genes_90 <- colnames(ctr_values)[ctr_values[901,] < 0.05]

library(VennDiagram)


draw.pairwise.venn(length(top_genes), length(top_genes_90), 
                   length(intersect(top_genes, top_genes_90)), c("60k", "70k"),col=c("green","red"),scaled=F) 












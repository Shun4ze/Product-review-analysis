# Chargement des bibliothèques
library(dplyr)
library(ggplot2)
library(readr)
library(tidytext)
library(SnowballC)
library(tm)
library(caTools)
library(cluster)
library(tidyr)
library(stringr)

# Chargement des données avec vérification
file_path <- "C:/Users/Chloé/Documents/COURS/ESME/INGE3/Data science sous R/projet/dataset/Reviews.csv"
if (!file.exists(file_path)) {
  stop("Erreur : le fichier spécifié n'existe pas. Vérifiez le chemin d'accès.")
}
data <- read.csv(file_path)

# Exploration initiale des données
str(data)
summary(data)
head(data)

# Calcul du ratio d'utilité
data <- data %>%
  mutate(HelpfulnessRatio = ifelse(HelpfulnessDenominator > 0, HelpfulnessNumerator / HelpfulnessDenominator, NA))

# Filtrage des valeurs NA dans la colonne HelpfulnessRatio
data_filtered <- data %>%
  filter(!is.na(HelpfulnessRatio))

# Visualisation de la distribution du ratio d’utilité
ggplot(data_filtered, aes(x = HelpfulnessRatio)) +
  geom_histogram(bins = 50, fill = "blue", alpha = 0.7) +
  labs(title = "Distribution du ratio d'utilité", x = "Ratio d'utilité", y = "Nombre d'avis")

# Calcul du ratio d'utilité moyen par score
avg_helpfulness_by_score <- data_filtered %>%
  group_by(Score) %>%
  summarise(Mean_HelpfulnessRatio = mean(HelpfulnessRatio, na.rm = TRUE))

# Visualisation du ratio d'utilité moyen par score
ggplot(avg_helpfulness_by_score, aes(x = factor(Score), y = Mean_HelpfulnessRatio)) +
  geom_bar(stat = "identity", fill = "coral") +
  labs(title = "Ratio d'utilité moyen par score", x = "Score", y = "Ratio d'utilité moyen")

# Création d'une colonne de sentiment
data_filtered <- data_filtered %>%
  mutate(Sentiment = ifelse(Score >= 4, "Positif", ifelse(Score <= 2, "Négatif", NA))) %>%
  filter(!is.na(Sentiment))

# Préparation des avis utiles (ratio > 0.8) pour l'analyse textuelle
useful_reviews <- data_filtered %>%
  filter(HelpfulnessRatio > 0.5) %>%
  sample_frac(0.2, replace = FALSE)

# Tokenisation, suppression des mots vides, et racinisation
tokens <- useful_reviews %>%
  unnest_tokens(word, Summary) %>%
  anti_join(stop_words, by = "word") %>%
  filter(!word %in% c("br"), !str_detect(word, "^[0-9]+$")) %>%
  mutate(word = wordStem(word, language = "en"))

# Vectorisation TF-IDF
tokens_vectorized <- tokens %>%
  count(Id, word, sort = TRUE) %>%
  bind_tf_idf(word, Id, n) %>%
  arrange(desc(tf_idf))

# Clustering pour identifier les thèmes récurrents (K-means)
terms_matrix <- tokens_vectorized %>%
  select(Id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) 
row.names(terms_matrix) <- terms_matrix$Id
terms_matrix <- as.matrix(terms_matrix[, -1])

#Standardisation de la matrice
terms_matrix <- scale(terms_matrix)

# Détermination du nombre optimal de clusters avec la méthode du coude
wss <- sapply(1:15, function(k) {
  kmeans(terms_matrix, centers = k, nstart = 25)$tot.withinss
})

# Visualisation de la méthode du coude
elbow_plot <- data.frame(k = 1:15, wss = wss)
ggplot(elbow_plot, aes(x = k, y = wss)) +
  geom_line() +
  geom_point() +
  labs(title = "Méthode du coude pour déterminer le nombre optimal de clusters",
       x = "Nombre de clusters (k)",
       y = "Inertie intra-cluster (WSS)") +
  theme_minimal()

# Détermination du nombre optimal de clusters avec l'indice de silhouette
avg_silhouette <- vector()
for (k in 2:15) {
  set.seed(123)
  kmeans_model <- kmeans(terms_matrix, centers = k, nstart = 25)
  silhouette_avg <- silhouette(kmeans_model$cluster, dist(terms_matrix))
  avg_silhouette[k] <- mean(silhouette_avg[, 3])  # Stocker la moyenne des silhouettes
  # Vérifier si la sortie de silhouette n'est pas NULL
  if (!is.null(silhouette_avg) && is.matrix(silhouette_avg)) {
    avg_silhouette[k] <- mean(silhouette_avg[, 3])
  } else {
    avg_silhouette[k] <- NA  
  }
}

# Tracer le graphique de l'indice de silhouette moyen
silhouette_plot <- data.frame(Clusters = 2:15, Silhouette = avg_silhouette[2:15])
ggplot(silhouette_plot, aes(x = Clusters, y = Silhouette)) +
  geom_line() +
  geom_point() +
  ggtitle("Indice de Silhouette moyen pour chaque nombre de clusters") +
  xlab("Nombre de clusters") +
  ylab("Silhouette moyenne")

# Exécution du clustering K-means avec le nombre optimal de clusters
set.seed(123)
k <- 7  
kmeans_result <- kmeans(terms_matrix, centers = k, nstart = 25)

# Réduction de dimension avec PCA pour visualiser les clusters
pca_result <- prcomp(terms_matrix, center = TRUE, scale. = TRUE)
pca_data <- as.data.frame(pca_result$x)
pca_data$Cluster <- as.factor(kmeans_result$cluster)

# Visualisation des clusters en 2D
ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(title = "Visualisation des thèmes par cluster",
       x = "Composante principale 1",
       y = "Composante principale 2") +
  theme_minimal() +
  theme(legend.position = "right")

# Regroupement des données pour traitement
tokens_vectorized$Cluster <- kmeans_result$cluster[tokens_vectorized$Id]
useful_reviews$Sentiment_Num <- ifelse(useful_reviews$Sentiment == "Positif", 1, 0)

model_data <- tokens_vectorized %>%
  left_join(useful_reviews %>% 
  select(Id, Sentiment_Num), by = "Id") %>%
  group_by(Id) %>%
  summarise(Cluster = unique(Cluster), Sentiment_Num = unique(Sentiment_Num)) %>%
  ungroup()

# Calcul du pourcentage de commentaires positifs par cluster
theme_sentiments <- model_data %>%
  filter(!is.na(Cluster)) %>%
  group_by(Cluster) %>%
  summarise(
    Total_Comments = n(),
    Positive_Comments = sum(Sentiment_Num == 1)
  ) %>%
  mutate(Positive_Percentage = (Positive_Comments / Total_Comments) * 100)

# Visualisation des pourcentages de commentaires positifs par cluster
ggplot(theme_sentiments, aes(x = factor(Cluster), y = Positive_Percentage)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Pourcentage de commentaires positifs par thème",
       x = "Thème (Cluster)",
       y = "Pourcentage de commentaires positifs (%)") +
  theme_minimal()

# Split des données en ensembles d'entraînement et de test
set.seed(123)
split <- sample.split(model_data$Sentiment_Num, SplitRatio = 0.7)
train_data <- model_data[split, ]
test_data <- model_data[!split, ]

# Modèle de régression logistique pour prédire le sentiment selon les thèmes
model <- glm(Sentiment_Num ~ Cluster, data = train_data, family = binomial)
summary(model)
saveRDS(model, file = "sentiment_model.rds")

# Prédictions et évaluation
test_data$Predicted_Prob <- predict(model, newdata = test_data, type = "response")
test_data$Predicted_Sentiment <- ifelse(test_data$Predicted_Prob >= 0.5, 1, 0)

# Matrice de confusion et précision du modèle
confusion_matrix <- table(test_data$Sentiment_Num, test_data$Predicted_Sentiment)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Précision du modèle:", accuracy, "\n")

# Visualisation de la matrice de confusion
confusion_melted <- as.data.frame(confusion_matrix)
names(confusion_melted) <- c("Réel", "Prédit", "Fréquence")

ggplot(confusion_melted, aes(x = Réel, y = Prédit, fill = Fréquence)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "red", high = "blue") +
  labs(title = "Matrice de confusion", x = "Sentiment réel", y = "Sentiment prédit") +
  geom_text(aes(label = Fréquence), color = "black") +
  theme_minimal()

# Utilisation du model

# Charger le modèle sauvegardé
model <- readRDS("sentiment_model.rds")

# Fonction pour prédire le sentiment d'un nouveau commentaire
predict_sentiment <- function(new_comment, model, stop_words, kmeans_result, original_terms) {
  new_data <- data.frame(Text = new_comment, stringsAsFactors = FALSE)
  
  # Traitement du nouveau commentaire pour convenir au model
  tokens_new <- new_data %>%
    unnest_tokens(word, Text) %>%
    anti_join(stop_words, by = "word") %>%
    filter(!word %in% c("br"), !grepl("^[0-9]+$", word)) %>%
    mutate(word = wordStem(word, language = "en"))
  
  tokens_vectorized_new <- tokens_new %>%
    count(Id = row_number(), word, sort = TRUE) %>%
    bind_tf_idf(word, Id, n) %>%
    arrange(desc(tf_idf))
  
  terms_matrix_new <- tokens_vectorized_new %>%
    pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0)
  
  # Alignement avec la matrice de termes originale
  missing_terms <- setdiff(colnames(original_terms), colnames(terms_matrix_new))
  for (term in missing_terms) {
    terms_matrix_new[[term]] <- 0
  }
  
  terms_matrix_new <- terms_matrix_new[, colnames(original_terms)]
  terms_matrix_new <- as.matrix(terms_matrix_new[, -1])
  
  distances <- apply(kmeans_result$centers, 1, function(center) {
    sum((terms_matrix_new - center) ^ 2)
  })
  
  # Attribution au cluster le plus proche
  new_cluster <- which.min(distances)
  
  # Prédiction du sentiment avec le modèle
  predicted_prob <- predict(model, newdata = data.frame(Cluster = new_cluster), type = "response")
  predicted_sentiment <- ifelse(predicted_prob >= 0.5, "Positif", "Négatif")
  
  return(predicted_sentiment)
}

data("stop_words")

# Prédire le sentiment pour un nouveau commentaire
new_comment <- "j'adore ce nouveau produit"
predicted_sentiment <- predict_sentiment(new_comment, model, stop_words, kmeans_result, terms_matrix)
cat("Sentiment prédit pour le commentaire : ",new_comment," est ", predicted_sentiment, "\n")

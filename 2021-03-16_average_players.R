# Libraries --------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(stacks)

# Download data ----------------------------------------------------------------

data <- tidytuesdayR::tt_load('2021-03-16')
old <- tidytuesdayR::tt_load('2019-07-30')

url <- "https://raw.githubusercontent.com/lizawood/apps-and-games/master/PC_Games/PCgames_2004_2018_raw.csv"

# read in raw data
price <- url %>%
  read_csv(col_types = "dcccccccc") %>%
  pull(Price)

# Cleanup ----------------------------------------------------------------------

price[price == "Free"] <- "0"
price <- as.numeric(price)

old$video_games$price <- price

games <- data$games %>%
  left_join(old$video_games, by = c("gamename" = "game")) %>%
  drop_na() %>%
  separate(owners, into = c("owners_low", "owners_high"), sep = "\\.\\.", convert = TRUE) %>%
  transmute(game = gamename,
            month = lubridate::ym(paste(year, month)),
            avg_players = avg,
            price,
            release_date = lubridate::mdy(release_date),
            publisher = map(publisher, str_split, pattern = ","),
            publisher = map_chr(publisher, ~ .x[[1]][1]),
            publisher = case_when(
              str_starts(publisher, "Bethesda") ~ "Bethesda",
              str_starts(publisher, "Ubisoft") ~ "Ubisoft",
              publisher == "Xbox Game Studios" ~ "Microsoft Studios",
              publisher == "U-Play Online" ~ "Ubisoft",
              TRUE ~ publisher
            ),
            owners_low = parse_number(owners_low),
            owners_high = parse_number(owners_high),
            owners = (owners_high + owners_low) / 2,
            metascore,
            time = as.numeric(month - release_date)
            ) %>%
  filter(time >= 0)

publisher_mc <- games %>%
  distinct(game, publisher, metascore) %>%
  group_by(publisher) %>%
  summarise(pub_games = n(), pub_mc = mean(metascore))

games <- games %>%
  left_join(publisher_mc, by = "publisher") %>%
  select(-game, -month, -release_date, -publisher, -owners_high, -owners_low)

# Modelling --------------------------------------------------------------------

set.seed(21316)

games_split <- initial_split(games)

games_train <- training(games_split)
games_test <- testing(games_split)

games_folds <- vfold_cv(games_train, v = 15)

base_rec <- recipe(avg_players ~ ., data = games_train) %>%
  step_sqrt(time, price) %>%
  step_log(owners) %>%
  step_normalize(all_predictors())

# Linear -----------------------------------------------------------------------

lin_model <- linear_reg() %>%
  set_engine("lm")

lin_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(lin_model)

lin_res <- fit_resamples(
  lin_wf,
  resamples = games_folds,
  control = control_stack_resamples()
)

# Decision Tree ----------------------------------------------------------------

dec_model <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_mode("regression") %>%
  set_engine("rpart")

dec_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(dec_model)

dec_res <- tune_grid(
  dec_wf,
  resamples = games_folds,
  grid = 30,
  control = control_stack_grid()
)

# Random Forest ----------------------------------------------------------------

rand_model <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("regression") %>%
  set_engine("ranger")

rand_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(rand_model)

rand_res <- tune_grid(
  rand_wf,
  resamples = games_folds,
  grid = 10,
  control = control_stack_grid()
)

# Stacks -----------------------------------------------------------------------

ensemble <- stacks() %>%
  add_candidates(lin_res) %>%
  add_candidates(dec_res) %>%
  add_candidates(rand_res) %>%
  blend_predictions() %>%
  fit_members()

# Vizualisation

bind_cols(
  games_test,
  predict(ensemble, games_test)
)  %>%
  ggplot(aes(avg_players, .pred)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  scale_x_continuous("Average player count") +
  scale_y_continuous("Predicted player count") +
  coord_obs_pred() +
  ggtitle("Predicted player count for videogames on Steam") +
  theme_light() +
  theme(
    panel.border = element_blank(),
    axis.ticks = element_blank()
  )

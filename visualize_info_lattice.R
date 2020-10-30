setwd("~/projects/synergistic")
rm(list=ls())
library(tidyverse)

position_map = list(
  c(1,1,2,0),
  c(2,1,1,1),
  c(1,2,3,1),
  c(3,1,0,2),
  c(2,2,2,2),
  c(1,3,4,2),
  c(3,2,1,3),
  c(2,3,3,3),
  c(3,3,2,4)
) %>% data.frame() %>% t() %>% data.frame()
names(position_map) = c("x_cardinality", "y_cardinality", "x", "y")

position_mapping = function(x_cardinality, y_cardinality) {
  case_when(
    x_cardinality == 1 & y_cardinality == 1 ~ c(2,0),
    x_cardinality == 2 & y_cardinality == 1 ~ c(1,1),
    x_cardinality == 1 & y_cardinality == 2 ~ c(3,1),
    x_cardinality == 3 & y_cardinality == 1 ~ c(0,2),
    x_cardinality == 2 & y_cardinality == 2 ~ c(2,2),
    x_cardinality == 1 & y_cardinality == 3 ~ c(4,2),
    x_cardinality == 3 & y_cardinality == 2 ~ c(1,3),
    x_cardinality == 2 & y_cardinality == 3 ~ c(3,3),
    x_cardinality == 3 & y_cardinality == 3 ~ c(2,4)
  )
}

d = read_csv("survey_nonuniform_20200807.csv") %>% mutate(i=1:nrow(.))

di = d %>% 
  select(-1:-9) %>% 
  gather(key, value, -mapping, -i, -is_systematic) 

di %>% 
  group_by(i) %>% 
    summarise(total=sum(value)) %>% 
    ungroup() # Check all equal

dm = d %>%
  select(1:9, mapping, is_systematic, i) %>%
  gather(key, value, -mapping, -is_systematic, -i) %>%
  separate(key, into=c("remove", "x_cardinality", "y_cardinality")) %>%
  select(-remove) %>%
  mutate(x_cardinality=as.numeric(x_cardinality),
         y_cardinality=as.numeric(y_cardinality))

# Histogram of spectrum_1_1 across all codes
dm %>%
  filter(x_cardinality == 1, y_cardinality == 1) %>%
  ggplot(aes(x=3-value, fill=is_systematic)) + geom_histogram(bins=20) + xlab("I[G:X]-sum_{ij} I[G_i : X_j]") + ylab("Number of codes") + theme_minimal()

to_string = function(x_cardinality, y_cardinality) {
  str_c(x_cardinality, "g ", y_cardinality, "x")
}

# Lattice
dm %>%
  filter(i %in% c(1,2,3,4)) %>%
  inner_join(position_map) %>%
  mutate(which=to_string(x_cardinality, y_cardinality)) %>%
  ggplot(aes(x=x, y=y, fill=value, label=which)) + 
    geom_point(shape=21, size=30) + 
    geom_text() +
    scale_fill_gradient2(low="red", high="blue", mid="white", midpoint=0, limits=c(-3.1, 3.1)) +
    xlim(-1, 5) +
    ylim(-1, 5) +
    theme_void() +
    facet_wrap(~mapping) +
    xlab("") + ylab("")

# Lattice
dm %>%
  filter(i %in% c(1,127,4628,20665)) %>%
  inner_join(position_map) %>%
  mutate(which=to_string(x_cardinality, y_cardinality)) %>%
  ggplot(aes(x=x, y=y, fill=value, label=which)) + 
    geom_point(shape=21, size=20) + 
    geom_text() +
    scale_fill_gradient2(low="red", high="blue", mid="white", midpoint=0, limits=c(-3.2, 3.2)) +
    xlim(-1, 5) +
    ylim(-1, 5) +
    theme_void() +
    facet_wrap(~mapping) +
    xlab("") + ylab("")


  
  
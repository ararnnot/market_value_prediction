library(tidyverse)
library(worldfootballR)

years       =   c(2019, 2020, 2021, 2022, 2023)
countries   =   c('Germany', 'England', 'Netherlands',
                  'Italy', 'Belgium', 'Spain',
                  'France', 'Portugal')

prices <- data.frame(
    country         =   character(),
    season_start    =   numeric(),
    info            =   character(),
    stringsAsFactors =  FALSE  
)

for(year in years){
    for(country in countries){
        print(paste0(country, ', ', year))
        table   <-  tm_player_market_values(country_name = country,
                                            start_year = year)
        data    <-  data.frame(country, year, table$player_name)
        prices  =   rbind(prices, data)
    }
}

write.csv(prices, 'data/prices_raw.csv', row.names = FALSE)


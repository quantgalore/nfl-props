# nfl-props
NFL Reception Player Props - [The Quant's Playbook](https://quantgalore.substack.com/) @ Substack | Quant Galore

This repository contains the necessary code for deploying a system that captures a theoretical edge in NFL reception player prop markets.

For ease of access, all functionality can be found in the "nfl-receptions-prediction" file, but this requires constantly re-building the dataset and training the models with each run.

It is recommend that you break this up into a separate dataset builder file which builds the initial dataset once then stores it in a SQL database. From this, you can quickly update new records. 

## Requirements
You will need a [Prop-Odds](https://www.prop-odds.com/?ref=quantgalore) API Key to pull the necessary market line values. 

## Methodology

For the full methodology, considerations, and best practices, refer to [The Quant's Playbook](https://quantgalore.substack.com/) at quantgalore.substack.com

The accompanying post is titled: "Sport Markets Aren't So Efficient After All... [Code Included]"

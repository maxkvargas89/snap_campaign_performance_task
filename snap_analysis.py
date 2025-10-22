#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 17:50:34 2025

@author: maxvargas
"""

# Import data

import pandas as pd
import numpy as np

df_performance = pd.read_csv('snap_performance_data.csv')
df_creative = pd.read_csv('snap_creative_library.csv')

# Basic Info About the Dataset

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

# Shape: rows x columns
print(f"\nDataset Shape: {df_performance.shape[0]:,} rows × {df_performance.shape[1]} columns")

# Column names and types
print(f"\nColumn Names and Data Types:")
print(df_performance.dtypes)

# Memory usage
print(f"\nMemory Usage: {df_performance.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# First few rows
print("\nFirst 5 Rows:")
print(df_performance.head())

# Last few rows (check if data looks consistent)
print("\nLast 5 Rows:")
print(df_performance.tail())

# Check for Missing Data

print("\n" + "=" * 60)
print("MISSING DATA ANALYSIS")
print("=" * 60)

# Count and percentage of missing values per column
missing = pd.DataFrame({
    'Column': df_performance.columns,
    'Missing_Count': df_performance.isnull().sum(),
    'Missing_Percent': (df_performance.isnull().sum() / len(df_performance) * 100).round(2)
})
missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing) > 0:
    print("\nColumns with Missing Values:")
    print(missing.to_string(index=False))
else:
    print("\nNo missing values found!")

# Summary Statistics

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Numeric columns only
print("\nNumeric Columns:")
print(df_performance.describe())

# Categorical columns: value counts
print("\nCategorical Columns (first 3):")
categorical_cols = df_performance.select_dtypes(include=['object']).columns[:3]

for col in categorical_cols:
    print(f"\n{col} - Unique Values: {df_performance[col].nunique()}")
    print(df_performance[col].value_counts().head(10))  # Top 10 values

# Data Quality Checks

print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

# Check for duplicate rows
duplicates = df_performance.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates:,} ({duplicates/len(df_performance)*100:.2f}%)")

# Check numeric columns for outliers (using IQR method)
numeric_cols = df_performance.select_dtypes(include=[np.number]).columns

print("\nPotential Outliers (values beyond 1.5 * IQR):")
for col in numeric_cols:
    Q1 = df_performance[col].quantile(0.25)
    Q3 = df_performance[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_performance[(df_performance[col] < lower_bound) | (df_performance[col] > upper_bound)][col].count()
    if outliers > 0:
        print(f"  {col}: {outliers:,} outliers ({outliers/len(df_performance)*100:.2f}%)")

# Clean data
df_performance['Spend'] = df_performance['Spend'].str.replace('[$,]', '', regex=True).astype(float)
df_performance['Purchases Value'] = df_performance['Purchases Value'].str.replace('[$,]', '', regex=True).astype(float)
df_performance['Paid Impressions'] = df_performance['Paid Impressions'].str.replace('[,]', '', regex=True).astype(float)
        
# Questions

print("\n" + "=" * 30)
print("QUESTION ANSWERS")
print("=" * 30)

# Question 1

print("\n" + "-" * 30)
print("1. Calculate the total installs for the campaign")
print("-" * 30)

tot_installs = df_performance['Installs'].sum()
print(tot_installs)

# Question 2

print("\n" + "-" * 30)
print("2. Calculate the cost per install for Jan 7 2022")
print("-" * 30)

df_performance_010722 = df_performance[df_performance['Day'] == '1/7/2022']
spend = df_performance_010722['Spend'].sum()
installs = df_performance_010722['Installs'].sum()
cost_per_install = spend / installs
print(f"${cost_per_install:.2f}")

# Question 3

print("\n" + "-" * 30)
print("3. Which day had the lowest CPM?")
print("-" * 30)

daily_summary_cpm = df_performance.groupby('Day').agg({
    'Spend': 'sum',
    'Paid Impressions': 'sum'
}).reset_index()

daily_summary_cpm['cpm'] = daily_summary_cpm['Spend']/(daily_summary_cpm['Paid Impressions']/1000)
daily_summary_cpm['cpm'] = daily_summary_cpm['cpm'].round(2)
daily_summary_cpm['cpm_rank'] = daily_summary_cpm['cpm'].rank(ascending=True, method='min')
print(daily_summary_cpm[daily_summary_cpm['cpm_rank'] == 1.0])

# Question 4

print("\n" + "-" * 30)
print("4. Did the day with most purchases have the highest ROAS")
print("-" * 30)

daily_summary_roas = df_performance.groupby('Day').agg({
    'Purchases': 'sum',
    'Purchases Value': 'sum',
    'Spend': 'sum'
}).reset_index()

daily_summary_roas['roas'] = daily_summary_roas['Purchases Value']/daily_summary_roas['Spend']
daily_summary_roas['roas'] = daily_summary_roas['roas'].round(2)
daily_summary_roas['roas_rank'] = daily_summary_roas['roas'].rank(ascending=False, method='min')
daily_summary_roas['purchases_rank'] = daily_summary_roas['Purchases'].rank(ascending=False, method='min')
print(daily_summary_roas[daily_summary_roas['purchases_rank'] == 1.0])
print(daily_summary_roas)

# Question 5

print("\n" + "-" * 30)
print("5. What was the ROAS for Ads that had Boots?")
print("-" * 30)

df_merged = df_performance.merge(df_creative, on='Creative ID', how='left')
df_merged_boots = df_merged[df_merged['Creative Name (Featured Product_CTA)'].str.contains('Boots', case=False, na=False)] 

boots_purchases_value = df_merged_boots['Purchases Value'].sum()
boots_spend_value = df_merged_boots['Spend'].sum()
boots_roas = boots_purchases_value/boots_purchases_value
print(boots_roas.round(2))

# Question 6 

print("\n" + "-" * 30)
print("6. Which combo of product featured and CTA had the most purchases?")
print("-" * 30)

creative_combo_summary = df_merged.groupby('Creative Name (Featured Product_CTA)').agg({
    'Purchases': 'sum'
}).reset_index()

creative_combo_summary['purchases_rank'] = creative_combo_summary['Purchases'].rank(ascending=False, method='min')
print(creative_combo_summary[creative_combo_summary['purchases_rank'] == 1.0])


# Trends and Observations

# Overall campaign ROAS
total_roas = df_merged['Purchases Value'].sum() / df_merged['Spend'].sum()

# ROAS by Product Featured
df_merged[['Product Featured', 'CTA']] = df_merged['Creative Name (Featured Product_CTA)'].str.split('_', n=1, expand=True)
roas_by_product = df_merged.groupby('Product Featured').agg({
    'Spend': 'sum',
    'Purchases Value': 'sum'
})
roas_by_product['ROAS'] = roas_by_product['Purchases Value'] / roas_by_product['Spend']

# ROAS by CTA
roas_by_cta = df_merged.groupby('CTA').agg({
    'Spend': 'sum',
    'Purchases Value': 'sum'
})
roas_by_cta['ROAS'] = roas_by_cta['Purchases Value'] / roas_by_cta['Spend']

# Daily performance
daily_perf = df_merged.groupby('Day').agg({
    'Spend': 'sum',
    'Purchases': 'sum',
    'Purchases Value': 'sum',
    'Installs': 'sum'
})
daily_perf['ROAS'] = daily_perf['Purchases Value'] / daily_perf['Spend']
daily_perf['CPI'] = daily_perf['Spend'] / daily_perf['Installs']

print(roas_by_product.sort_values('ROAS', ascending=False))
print(roas_by_cta.sort_values('ROAS', ascending=False))
print(daily_perf.sort_values('ROAS', ascending=False))
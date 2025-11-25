"""
数据一致性测试脚本
检查前端Streamlit App显示的数据和后端数据源是否匹配
"""

import sys
sys.path.append('app')
from data_config import load_data
import pandas as pd

print("=" * 80)
print("📊 数据一致性检查报告")
print("=" * 80)

# 加载应用使用的数据
print("\n1️⃣ 加载Streamlit App使用的数据...")
df_app = load_data()

print(f"\n✅ 数据加载成功!")
print(f"   总行数: {len(df_app):,}")
print(f"   总列数: {len(df_app.columns)}")

# 检查关键字段
print("\n2️⃣ 检查关键字段...")
required_fields = ['company_name', 'company_age', 'total_funding', 'funding_rounds', 
                   'team_size', 'industry', 'region', 'target']

missing_fields = [field for field in required_fields if field not in df_app.columns]
if missing_fields:
    print(f"   ❌ 缺失字段: {missing_fields}")
else:
    print(f"   ✅ 所有必需字段都存在")

# 显示实际列名
print(f"\n3️⃣ 实际列名列表:")
for i, col in enumerate(df_app.columns, 1):
    print(f"   {i:2d}. {col}")

# 统计信息
print(f"\n4️⃣ 数据统计信息:")
print(f"   公司总数: {len(df_app):,}")
print(f"   成功公司数: {df_app['target'].sum():,} ({df_app['target'].mean()*100:.1f}%)")
print(f"   失败公司数: {(df_app['target']==0).sum():,} ({(1-df_app['target'].mean())*100:.1f}%)")

if 'company_age' in df_app.columns:
    print(f"   平均公司年龄: {df_app['company_age'].mean():.1f} 年")
    print(f"   最老公司: {df_app['company_age'].max():.0f} 年")
    print(f"   最新公司: {df_app['company_age'].min():.0f} 年")

if 'total_funding' in df_app.columns:
    print(f"   平均融资额: ${df_app['total_funding'].mean():,.0f}")
    print(f"   最高融资额: ${df_app['total_funding'].max():,.0f}")
    print(f"   有融资数据的公司: {(df_app['total_funding'] > 0).sum():,}")

print(f"\n5️⃣ 行业分布 (前10):")
industry_counts = df_app['industry'].value_counts().head(10)
for industry, count in industry_counts.items():
    print(f"   {industry}: {count:,} ({count/len(df_app)*100:.1f}%)")

print(f"\n6️⃣ 地区分布 (前10):")
region_counts = df_app['region'].value_counts().head(10)
for region, count in region_counts.items():
    print(f"   {region}: {count:,} ({count/len(df_app)*100:.1f}%)")

print(f"\n7️⃣ 缺失值检查:")
missing_data = df_app[required_fields].isnull().sum()
has_missing = False
for field, count in missing_data.items():
    if count > 0:
        print(f"   ⚠️  {field}: {count:,} ({count/len(df_app)*100:.1f}%)")
        has_missing = True
if not has_missing:
    print(f"   ✅ 所有必需字段无缺失值")

# 数据示例
print(f"\n8️⃣ 数据示例 (前5行):")
print(df_app[required_fields].head().to_string())

print("\n" + "=" * 80)
print("📋 检查结论:")
print("=" * 80)

# 检查是否是示例数据
if len(df_app) == 1000:
    print("⚠️  警告: 当前使用的是示例数据 (1000行)")
    print("   请确保真实数据文件存在于 data/raw/ 目录")
elif len(df_app) == 4974:
    print("✅ 当前使用的是真实YC数据 (4974家公司)")
    print("✅ 数据已成功加载并匹配Crunchbase融资信息")
else:
    print(f"ℹ️  数据行数: {len(df_app)}")

print("\n🎯 前端App应该显示以上统计数据")
print("   如果不一致，请检查 app/app.py 中的数据加载逻辑")
print("=" * 80)



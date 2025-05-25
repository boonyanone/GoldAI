#!/usr/bin/env python3
"""
Phase 1: Data Validation Framework
Validates Cup & Handle pattern quality before model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import json
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import seaborn as sns
from collections import Counter

class PatternValidator:
    """
    Comprehensive validation for Cup & Handle patterns
    """
    
    def __init__(self, data_dir='data/raw/crypto_patterns'):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        
    def load_patterns_and_data(self):
        """Load all pattern data for validation"""
        patterns_by_symbol = {}
        
        for pattern_file in self.data_dir.glob('*_pattern_candidates.json'):
            symbol = pattern_file.stem.replace('_4h_pattern_candidates', '').replace('_1d_pattern_candidates', '')
            timeframe = '4h' if '_4h_' in pattern_file.stem else '1d'
            
            # Load patterns
            with open(pattern_file, 'r') as f:
                patterns = json.load(f)
            
            # Load corresponding data
            data_files = list(self.data_dir.glob(f"{symbol}_{timeframe}_*.csv"))
            if data_files:
                data = pd.read_csv(data_files[0], index_col=0, parse_dates=True)
                
                patterns_by_symbol[f"{symbol}_{timeframe}"] = {
                    'patterns': patterns,
                    'data': data,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
        
        return patterns_by_symbol
    
    def validate_pattern_statistics(self, patterns_data):
        """Statistical validation of pattern characteristics"""
        print("🔍 Phase 1: Statistical Pattern Validation")
        print("=" * 60)
        
        all_patterns = []
        for symbol_data in patterns_data.values():
            all_patterns.extend(symbol_data['patterns'])
        
        if not all_patterns:
            print("❌ No patterns found for validation")
            return None
        
        # Extract key metrics
        durations = [p['length'] for p in all_patterns]
        returns = [p['price_action']['total_return'] * 100 for p in all_patterns]
        drawdowns = [p['price_action']['max_drawdown'] * 100 for p in all_patterns]
        volatilities = [p['volatility_analysis']['avg_volatility'] for p in all_patterns]
        
        stats = {
            'total_patterns': len(all_patterns),
            'duration_stats': {
                'mean': np.mean(durations),
                'median': np.median(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations)
            },
            'return_stats': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'positive_ratio': sum(1 for r in returns if r > 0) / len(returns),
                'above_5pct': sum(1 for r in returns if r > 5) / len(returns)
            },
            'drawdown_stats': {
                'mean': np.mean(drawdowns),
                'median': np.median(drawdowns),
                'above_50pct': sum(1 for d in drawdowns if d > 50) / len(drawdowns)
            }
        }
        
        # Print statistics
        print(f"📊 Total Patterns: {stats['total_patterns']:,}")
        print(f"\n📏 Duration Analysis:")
        print(f"   Mean: {stats['duration_stats']['mean']:.1f} periods")
        print(f"   Median: {stats['duration_stats']['median']:.1f} periods")
        print(f"   Range: {stats['duration_stats']['min']:.0f} - {stats['duration_stats']['max']:.0f}")
        
        print(f"\n💰 Return Analysis:")
        print(f"   Mean: {stats['return_stats']['mean']:.2f}%")
        print(f"   Median: {stats['return_stats']['median']:.2f}%")
        print(f"   Positive Ratio: {stats['return_stats']['positive_ratio']:.1%}")
        print(f"   Above 5%: {stats['return_stats']['above_5pct']:.1%}")
        
        print(f"\n📉 Risk Analysis:")
        print(f"   Mean Drawdown: {stats['drawdown_stats']['mean']:.1f}%")
        print(f"   High Risk (>50% DD): {stats['drawdown_stats']['above_50pct']:.1%}")
        
        return stats
    
    def create_validation_charts(self, patterns_data):
        """Create comprehensive validation charts"""
        all_patterns = []
        for symbol_data in patterns_data.values():
            all_patterns.extend(symbol_data['patterns'])
        
        if not all_patterns:
            return
        
        # Extract metrics
        durations = [p['length'] for p in all_patterns]
        returns = [p['price_action']['total_return'] * 100 for p in all_patterns]
        drawdowns = [p['price_action']['max_drawdown'] * 100 for p in all_patterns]
        volatilities = [p['volatility_analysis']['avg_volatility'] for p in all_patterns]
        
        # Create multi-panel validation chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 1: Pattern Validation Analysis', fontsize=16, fontweight='bold')
        
        # Duration distribution
        axes[0,0].hist(durations, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].set_title('Pattern Duration Distribution')
        axes[0,0].set_xlabel('Duration (periods)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(np.median(durations), color='red', linestyle='--', label=f'Median: {np.median(durations):.1f}')
        axes[0,0].legend()
        
        # Return distribution
        axes[0,1].hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].set_title('Pattern Return Distribution')
        axes[0,1].set_xlabel('Return (%)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(0, color='black', linestyle='-', alpha=0.5)
        axes[0,1].axvline(5, color='red', linestyle='--', label='5% Target')
        axes[0,1].legend()
        
        # Drawdown distribution
        axes[0,2].hist(drawdowns, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0,2].set_title('Maximum Drawdown Distribution')
        axes[0,2].set_xlabel('Max Drawdown (%)')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].axvline(50, color='black', linestyle='--', label='50% Risk Level')
        axes[0,2].legend()
        
        # Return vs Duration scatter
        axes[1,0].scatter(durations, returns, alpha=0.5, s=20)
        axes[1,0].set_title('Return vs Duration')
        axes[1,0].set_xlabel('Duration (periods)')
        axes[1,0].set_ylabel('Return (%)')
        axes[1,0].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[1,0].axhline(5, color='red', linestyle='--', alpha=0.5)
        
        # Risk-Return scatter
        axes[1,1].scatter(drawdowns, returns, alpha=0.5, s=20, c=volatilities, cmap='viridis')
        axes[1,1].set_title('Risk-Return Profile')
        axes[1,1].set_xlabel('Max Drawdown (%)')
        axes[1,1].set_ylabel('Return (%)')
        axes[1,1].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[1,1].axvline(50, color='red', linestyle='--', alpha=0.3)
        
        # Quality score distribution
        quality_scores = [self.calculate_quality_score(p) for p in all_patterns]
        axes[1,2].hist(quality_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1,2].set_title('Pattern Quality Score Distribution')
        axes[1,2].set_xlabel('Quality Score (0-100)')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].axvline(70, color='red', linestyle='--', label='A Grade (70+)')
        axes[1,2].axvline(50, color='orange', linestyle='--', label='B Grade (50+)')
        axes[1,2].legend()
        
        plt.tight_layout()
        
        # Save validation chart
        output_dir = Path('data/validation')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'phase1_pattern_validation.png', dpi=300, bbox_inches='tight')
        print(f"💾 Validation chart saved: {output_dir / 'phase1_pattern_validation.png'}")
        
        return fig
    
    def calculate_quality_score(self, pattern):
        """Calculate quality score for a pattern (0-100)"""
        score = 0
        
        # Duration score (20 points)
        duration = pattern['length']
        if 20 <= duration <= 200:
            score += 20
        elif 10 <= duration <= 300:
            score += 10
        
        # Return score (25 points)
        total_return = pattern['price_action']['total_return']
        if total_return > 0.10:  # >10%
            score += 25
        elif total_return > 0.05:  # >5%
            score += 20
        elif total_return > 0:  # Positive
            score += 10
        
        # Drawdown score (20 points)
        drawdown = pattern['price_action']['max_drawdown']
        if drawdown < 0.20:  # <20%
            score += 20
        elif drawdown < 0.40:  # <40%
            score += 15
        elif drawdown < 0.60:  # <60%
            score += 10
        
        # Volume pattern score (15 points)
        if pattern['volume_pattern']['volume_increase_late']:
            score += 10
        if pattern['volume_pattern']['volume_spike_end']:
            score += 5
        
        # Volatility score (10 points)
        if pattern['volatility_analysis']['volatility_decreasing']:
            score += 10
        
        # Recovery score (10 points)
        recovery_ratio = pattern['price_action']['recovery_ratio']
        if recovery_ratio > 0.8:
            score += 10
        elif recovery_ratio > 0.6:
            score += 7
        elif recovery_ratio > 0.4:
            score += 5
        
        return min(score, 100)
    
    def create_sample_patterns_for_review(self, patterns_data, num_samples=20):
        """Create sample pattern visualizations for manual review"""
        print(f"\n🎯 Creating {num_samples} sample patterns for manual review...")
        
        # Collect all patterns with quality scores
        all_patterns_with_scores = []
        for symbol_key, symbol_data in patterns_data.items():
            for pattern in symbol_data['patterns']:
                quality_score = self.calculate_quality_score(pattern)
                all_patterns_with_scores.append({
                    'pattern': pattern,
                    'data': symbol_data['data'],
                    'symbol': symbol_data['symbol'],
                    'timeframe': symbol_data['timeframe'],
                    'quality_score': quality_score
                })
        
        # Sort by quality score and select diverse samples
        all_patterns_with_scores.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Select samples: high quality, medium quality, low quality
        high_quality = all_patterns_with_scores[:num_samples//3]
        mid_start = len(all_patterns_with_scores) // 3
        medium_quality = all_patterns_with_scores[mid_start:mid_start + num_samples//3]
        low_quality = all_patterns_with_scores[-num_samples//3:]
        
        samples = high_quality + medium_quality + low_quality
        
        # Create sample validation directory
        sample_dir = Path('data/validation/sample_patterns')
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate individual pattern charts
        for i, sample in enumerate(samples):
            fig = self.create_single_pattern_chart(sample, i+1)
            if fig:
                fig.savefig(sample_dir / f"pattern_{i+1:02d}_score_{sample['quality_score']:.0f}.png", 
                           dpi=200, bbox_inches='tight')
                plt.close(fig)
        
        # Create annotation template
        annotation_template = self.create_annotation_template(samples)
        with open(sample_dir / 'annotation_template.json', 'w') as f:
            json.dump(annotation_template, f, indent=2)
        
        print(f"✅ Sample patterns saved to: {sample_dir}")
        print(f"📝 Annotation template: {sample_dir / 'annotation_template.json'}")
        
        return samples
    
    def create_single_pattern_chart(self, sample, pattern_id):
        """Create a single pattern chart for review"""
        pattern = sample['pattern']
        data = sample['data']
        
        try:
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            
            # Get pattern data with context
            context_start = max(0, start_idx - 50)
            context_end = min(len(data), end_idx + 50)
            
            chart_data = data.iloc[context_start:context_end]
            pattern_data = data.iloc[start_idx:end_idx]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Price chart
            ax1.plot(chart_data.index, chart_data['Close'], color='blue', linewidth=1, alpha=0.7)
            ax1.plot(pattern_data.index, pattern_data['Close'], color='red', linewidth=2, label='Pattern')
            
            # Highlight pattern area
            ax1.axvspan(pattern_data.index[0], pattern_data.index[-1], alpha=0.2, color='yellow')
            
            # Add moving averages if available
            if 'SMA_20' in chart_data.columns:
                ax1.plot(chart_data.index, chart_data['SMA_20'], color='orange', alpha=0.6, linewidth=1)
            
            # Pattern info
            return_pct = pattern['price_action']['total_return'] * 100
            quality_score = sample['quality_score']
            
            title = (f"Pattern {pattern_id}: {sample['symbol']} ({sample['timeframe']}) | "
                    f"Score: {quality_score:.0f} | Return: {return_pct:.1f}% | "
                    f"Duration: {pattern['length']} periods")
            
            ax1.set_title(title, fontsize=12, fontweight='bold')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            volume_colors = ['green' if close >= open else 'red' 
                           for open, close in zip(chart_data['Open'], chart_data['Close'])]
            ax2.bar(chart_data.index, chart_data['Volume'], color=volume_colors, alpha=0.7)
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            # Highlight pattern volume
            pattern_volume = pattern_data['Volume']
            ax2.bar(pattern_data.index, pattern_volume, color='red', alpha=0.8)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating chart for pattern {pattern_id}: {e}")
            return None
    
    def create_annotation_template(self, samples):
        """Create annotation template for manual validation"""
        template = {
            'validation_metadata': {
                'created_date': datetime.now().isoformat(),
                'total_samples': len(samples),
                'annotation_instructions': {
                    'cup_criteria': 'U-shaped price decline and recovery lasting 15-200 periods',
                    'handle_criteria': 'Small consolidation near cup rim (optional)',
                    'breakout_criteria': 'Price breaks above cup rim with volume',
                    'quality_grades': {
                        'A': 'Perfect Cup & Handle (80-100 points)',
                        'B': 'Good pattern with minor flaws (60-79 points)', 
                        'C': 'Recognizable but imperfect (40-59 points)',
                        'D': 'Poor pattern quality (20-39 points)',
                        'F': 'Not a Cup & Handle pattern (0-19 points)'
                    }
                }
            },
            'patterns_to_review': []
        }
        
        for i, sample in enumerate(samples):
            pattern_entry = {
                'pattern_id': i + 1,
                'symbol': sample['symbol'],
                'timeframe': sample['timeframe'],
                'auto_quality_score': sample['quality_score'],
                'pattern_metrics': {
                    'duration': sample['pattern']['length'],
                    'return_pct': sample['pattern']['price_action']['total_return'] * 100,
                    'max_drawdown_pct': sample['pattern']['price_action']['max_drawdown'] * 100,
                    'recovery_ratio': sample['pattern']['price_action']['recovery_ratio']
                },
                'manual_annotation': {
                    'is_valid_cup_handle': None,  # True/False
                    'manual_quality_grade': None,  # A/B/C/D/F
                    'cup_quality': None,  # 1-10
                    'handle_quality': None,  # 1-10 (0 if no handle)
                    'volume_confirmation': None,  # True/False
                    'notes': '',
                    'recommended_action': None  # 'use_for_training', 'exclude', 'needs_review'
                }
            }
            template['patterns_to_review'].append(pattern_entry)
        
        return template
    
    def generate_validation_report(self, patterns_data):
        """Generate comprehensive validation report"""
        stats = self.validate_pattern_statistics(patterns_data)
        if not stats:
            return
        
        # Generate quality distribution
        all_patterns = []
        for symbol_data in patterns_data.values():
            all_patterns.extend(symbol_data['patterns'])
        
        quality_scores = [self.calculate_quality_score(p) for p in all_patterns]
        grade_distribution = {
            'A_grade': sum(1 for s in quality_scores if s >= 80),
            'B_grade': sum(1 for s in quality_scores if 60 <= s < 80),
            'C_grade': sum(1 for s in quality_scores if 40 <= s < 60),
            'D_grade': sum(1 for s in quality_scores if 20 <= s < 40),
            'F_grade': sum(1 for s in quality_scores if s < 20)
        }
        
        # Create validation report
        report = {
            'validation_date': datetime.now().isoformat(),
            'summary_statistics': stats,
            'quality_distribution': grade_distribution,
            'recommendations': self.generate_recommendations(stats, grade_distribution),
            'next_steps': [
                '1. Review sample patterns manually',
                '2. Annotate pattern quality',
                '3. Filter high-quality patterns for training',
                '4. Proceed to Phase 2 with validated data'
            ]
        }
        
        # Save report
        output_dir = Path('data/validation')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📋 Validation Report Summary:")
        print(f"   High Quality (A): {grade_distribution['A_grade']:,} patterns")
        print(f"   Good Quality (B): {grade_distribution['B_grade']:,} patterns")
        print(f"   Acceptable (C): {grade_distribution['C_grade']:,} patterns")
        print(f"   Poor Quality (D-F): {grade_distribution['D_grade'] + grade_distribution['F_grade']:,} patterns")
        
        total_usable = grade_distribution['A_grade'] + grade_distribution['B_grade']
        print(f"   Usable for Training: {total_usable:,} patterns ({total_usable/len(all_patterns):.1%})")
        
        return report
    
    def generate_recommendations(self, stats, grade_distribution):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        total_patterns = stats['total_patterns']
        high_quality = grade_distribution['A_grade'] + grade_distribution['B_grade']
        
        if high_quality < 100:
            recommendations.append("⚠️  Low number of high-quality patterns. Consider expanding data collection.")
        
        if stats['return_stats']['positive_ratio'] < 0.5:
            recommendations.append("⚠️  Less than 50% positive returns. Review pattern detection criteria.")
        
        if stats['return_stats']['above_5pct'] < 0.3:
            recommendations.append("⚠️  Low percentage of patterns with >5% returns. Consider stricter filtering.")
        
        if high_quality >= 500:
            recommendations.append("✅ Sufficient high-quality patterns for model training.")
        
        if stats['drawdown_stats']['above_50pct'] > 0.3:
            recommendations.append("⚠️  High percentage of risky patterns. Consider risk filtering.")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Phase 1: Pattern Data Validation')
    parser.add_argument('--data-dir', type=str, default='data/raw/crypto_patterns', help='Data directory')
    parser.add_argument('--sample-size', type=int, default=30, help='Number of sample patterns for manual review')
    parser.add_argument('--skip-charts', action='store_true', help='Skip chart generation for faster validation')
    
    args = parser.parse_args()
    
    print("🔍 Phase 1: Pattern Data Validation")
    print(f"📁 Data source: {args.data_dir}")
    print(f"📊 Sample size: {args.sample_size}")
    
    # Initialize validator
    validator = PatternValidator(data_dir=args.data_dir)
    
    # Load patterns and data
    print("\n📥 Loading pattern data...")
    patterns_data = validator.load_patterns_and_data()
    
    if not patterns_data:
        print("❌ No pattern data found. Run Phase 1 data collection first.")
        return
    
    print(f"✅ Loaded data for {len(patterns_data)} symbol/timeframe combinations")
    
    # Statistical validation
    stats = validator.validate_pattern_statistics(patterns_data)
    
    # Create validation charts
    if not args.skip_charts:
        print("\n📊 Creating validation charts...")
        validator.create_validation_charts(patterns_data)
    
    # Generate sample patterns for manual review
    print(f"\n🎯 Generating {args.sample_size} sample patterns...")
    samples = validator.create_sample_patterns_for_review(patterns_data, args.sample_size)
    
    # Generate validation report
    print("\n📋 Generating validation report...")
    report = validator.generate_validation_report(patterns_data)
    
    print("\n✅ Phase 1 validation complete!")
    print("📁 Check data/validation/ for:")
    print("   - validation_report.json (summary statistics)")
    print("   - phase1_pattern_validation.png (analysis charts)")
    print("   - sample_patterns/ (manual review samples)")
    print("\n🎯 Next steps:")
    print("   1. Review sample patterns in data/validation/sample_patterns/")
    print("   2. Complete manual annotation using annotation_template.json")
    print("   3. Filter high-quality patterns for Phase 2 training")

if __name__ == "__main__":
    main() 
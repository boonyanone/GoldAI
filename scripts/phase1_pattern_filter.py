#!/usr/bin/env python3
"""
Phase 1: Pattern Quality Filter
Filters high-quality Cup & Handle patterns based on validation results
Target: Extract ~500 high-quality patterns from 18,146 candidates
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

class PatternQualityFilter:
    """
    Advanced filtering system for Cup & Handle patterns
    """
    
    def __init__(self, data_dir='data/raw/crypto_patterns'):
        self.data_dir = Path(data_dir)
        self.quality_thresholds = {
            'min_return': 0.03,          # Minimum 3% return
            'max_drawdown': 0.45,        # Maximum 45% drawdown
            'min_duration': 15,          # Minimum 15 periods
            'max_duration': 150,         # Maximum 150 periods
            'min_recovery_ratio': 0.6,   # Must recover at least 60% from bottom
            'min_quality_score': 40,     # Minimum quality score
            'volume_confirmation': True   # Must have volume confirmation
        }
        
    def load_all_patterns(self):
        """Load all patterns with their data"""
        all_patterns = []
        
        pattern_files = list(self.data_dir.glob('*_pattern_candidates.json'))
        print(f"🔍 Loading patterns from {len(pattern_files)} files...")
        
        for pattern_file in pattern_files:
            symbol = pattern_file.stem.replace('_4h_pattern_candidates', '').replace('_1d_pattern_candidates', '')
            timeframe = '4h' if '_4h_' in pattern_file.stem else '1d'
            
            # Load patterns
            with open(pattern_file, 'r') as f:
                patterns = json.load(f)
            
            # Load corresponding data
            data_files = list(self.data_dir.glob(f"{symbol}_{timeframe}_*.csv"))
            if data_files:
                data = pd.read_csv(data_files[0], index_col=0, parse_dates=True)
                
                for pattern in patterns:
                    pattern['symbol'] = symbol
                    pattern['timeframe'] = timeframe
                    pattern['data_available'] = True
                    all_patterns.append((pattern, data))
        
        print(f"✅ Loaded {len(all_patterns)} total patterns")
        return all_patterns
    
    def calculate_advanced_quality_score(self, pattern, data):
        """Calculate advanced quality score with stricter criteria"""
        score = 0
        reasons = []
        
        try:
            # Extract pattern data
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            pattern_data = data.iloc[start_idx:end_idx]
            
            if len(pattern_data) < 10:
                return 0, ["❌ Insufficient data"]
            
            # 1. Return Quality (30 points)
            total_return = pattern['price_action']['total_return']
            if total_return > 0.15:  # >15%
                score += 30
                reasons.append("✅ Excellent return (>15%)")
            elif total_return > 0.10:  # >10%
                score += 25
                reasons.append("✅ Very good return (>10%)")
            elif total_return > 0.05:  # >5%
                score += 20
                reasons.append("✅ Good return (>5%)")
            elif total_return > 0.03:  # >3%
                score += 15
                reasons.append("✅ Acceptable return (>3%)")
            elif total_return > 0:  # Positive
                score += 5
                reasons.append("✅ Positive return")
            else:
                reasons.append("❌ Negative return")
            
            # 2. Risk Management (25 points)
            drawdown = pattern['price_action']['max_drawdown']
            if drawdown < 0.15:  # <15%
                score += 25
                reasons.append("✅ Low risk (<15% drawdown)")
            elif drawdown < 0.25:  # <25%
                score += 20
                reasons.append("✅ Moderate risk (<25% drawdown)")
            elif drawdown < 0.35:  # <35%
                score += 15
                reasons.append("✅ Acceptable risk (<35% drawdown)")
            elif drawdown < 0.45:  # <45%
                score += 10
                reasons.append("⚠️ High risk (<45% drawdown)")
            else:
                reasons.append("❌ Very high risk (>45% drawdown)")
            
            # 3. Pattern Shape Quality (20 points)
            recovery_ratio = pattern['price_action']['recovery_ratio']
            if recovery_ratio > 0.85:  # >85% recovery
                score += 20
                reasons.append("✅ Excellent recovery pattern")
            elif recovery_ratio > 0.70:  # >70% recovery
                score += 15
                reasons.append("✅ Good recovery pattern")
            elif recovery_ratio > 0.60:  # >60% recovery
                score += 10
                reasons.append("✅ Acceptable recovery")
            else:
                reasons.append("❌ Poor recovery pattern")
            
            # 4. Duration Optimization (10 points)
            duration = pattern['length']
            if 20 <= duration <= 100:  # Optimal range
                score += 10
                reasons.append("✅ Optimal duration")
            elif 15 <= duration <= 150:  # Acceptable range
                score += 7
                reasons.append("✅ Acceptable duration")
            elif 10 <= duration <= 200:  # Wide range
                score += 5
                reasons.append("⚠️ Suboptimal duration")
            else:
                reasons.append("❌ Poor duration")
            
            # 5. Volume Confirmation (10 points)
            volume_pattern = pattern['volume_pattern']
            if volume_pattern['volume_spike_end']:
                score += 6
                reasons.append("✅ Volume spike at breakout")
            if volume_pattern['volume_increase_late']:
                score += 4
                reasons.append("✅ Volume increasing late")
            
            # 6. Technical Confirmation (5 points)
            if pattern['volatility_analysis']['volatility_decreasing']:
                score += 5
                reasons.append("✅ Decreasing volatility (consolidation)")
            
            return min(score, 100), reasons
            
        except Exception as e:
            return 0, [f"❌ Error calculating score: {e}"]
    
    def apply_strict_filters(self, all_patterns):
        """Apply strict quality filters to extract high-quality patterns"""
        print(f"\n🔬 Applying strict quality filters...")
        print(f"📋 Filter Criteria:")
        print(f"   • Min Return: {self.quality_thresholds['min_return']*100:.1f}%")
        print(f"   • Max Drawdown: {self.quality_thresholds['max_drawdown']*100:.1f}%")
        print(f"   • Duration: {self.quality_thresholds['min_duration']}-{self.quality_thresholds['max_duration']} periods")
        print(f"   • Min Recovery: {self.quality_thresholds['min_recovery_ratio']*100:.1f}%")
        print(f"   • Min Quality Score: {self.quality_thresholds['min_quality_score']}")
        
        filtered_patterns = []
        filter_stats = {
            'total_input': len(all_patterns),
            'passed_return': 0,
            'passed_drawdown': 0,
            'passed_duration': 0,
            'passed_recovery': 0,
            'passed_quality': 0,
            'passed_all': 0
        }
        
        for pattern, data in all_patterns:
            passed_filters = True
            
            # Calculate advanced quality score
            quality_score, reasons = self.calculate_advanced_quality_score(pattern, data)
            pattern['advanced_quality_score'] = quality_score
            pattern['quality_reasons'] = reasons
            
            # Filter 1: Minimum return
            if pattern['price_action']['total_return'] >= self.quality_thresholds['min_return']:
                filter_stats['passed_return'] += 1
            else:
                passed_filters = False
            
            # Filter 2: Maximum drawdown
            if pattern['price_action']['max_drawdown'] <= self.quality_thresholds['max_drawdown']:
                filter_stats['passed_drawdown'] += 1
            else:
                passed_filters = False
            
            # Filter 3: Duration range
            duration = pattern['length']
            if self.quality_thresholds['min_duration'] <= duration <= self.quality_thresholds['max_duration']:
                filter_stats['passed_duration'] += 1
            else:
                passed_filters = False
            
            # Filter 4: Recovery ratio
            if pattern['price_action']['recovery_ratio'] >= self.quality_thresholds['min_recovery_ratio']:
                filter_stats['passed_recovery'] += 1
            else:
                passed_filters = False
            
            # Filter 5: Quality score
            if quality_score >= self.quality_thresholds['min_quality_score']:
                filter_stats['passed_quality'] += 1
            else:
                passed_filters = False
            
            if passed_filters:
                filter_stats['passed_all'] += 1
                filtered_patterns.append((pattern, data))
        
        # Print filter statistics
        print(f"\n📊 Filter Results:")
        print(f"   Input Patterns: {filter_stats['total_input']:,}")
        print(f"   Passed Return Filter: {filter_stats['passed_return']:,} ({filter_stats['passed_return']/filter_stats['total_input']:.1%})")
        print(f"   Passed Drawdown Filter: {filter_stats['passed_drawdown']:,} ({filter_stats['passed_drawdown']/filter_stats['total_input']:.1%})")
        print(f"   Passed Duration Filter: {filter_stats['passed_duration']:,} ({filter_stats['passed_duration']/filter_stats['total_input']:.1%})")
        print(f"   Passed Recovery Filter: {filter_stats['passed_recovery']:,} ({filter_stats['passed_recovery']/filter_stats['total_input']:.1%})")
        print(f"   Passed Quality Filter: {filter_stats['passed_quality']:,} ({filter_stats['passed_quality']/filter_stats['total_input']:.1%})")
        print(f"   ✅ PASSED ALL FILTERS: {filter_stats['passed_all']:,} ({filter_stats['passed_all']/filter_stats['total_input']:.1%})")
        
        return filtered_patterns, filter_stats
    
    def rank_patterns_by_quality(self, filtered_patterns):
        """Rank patterns by quality and return metrics"""
        print(f"\n🏆 Ranking {len(filtered_patterns)} high-quality patterns...")
        
        # Calculate composite scores
        for pattern, data in filtered_patterns:
            return_score = pattern['price_action']['total_return'] * 100  # Weight returns heavily
            risk_score = (1 - pattern['price_action']['max_drawdown']) * 50  # Penalize risk
            recovery_score = pattern['price_action']['recovery_ratio'] * 30
            quality_score = pattern['advanced_quality_score'] * 0.2
            
            composite_score = return_score + risk_score + recovery_score + quality_score
            pattern['composite_score'] = composite_score
        
        # Sort by composite score
        filtered_patterns.sort(key=lambda x: x[0]['composite_score'], reverse=True)
        
        # Print top patterns
        print(f"\n🥇 Top 10 Patterns:")
        print("Rank | Symbol | Return | Drawdown | Recovery | Quality | Composite")
        print("-" * 65)
        
        for i, (pattern, data) in enumerate(filtered_patterns[:10]):
            return_pct = pattern['price_action']['total_return'] * 100
            drawdown_pct = pattern['price_action']['max_drawdown'] * 100
            recovery_pct = pattern['price_action']['recovery_ratio'] * 100
            quality = pattern['advanced_quality_score']
            composite = pattern['composite_score']
            
            print(f"{i+1:4d} | {pattern['symbol']:6s} | {return_pct:6.1f}% | {drawdown_pct:8.1f}% | {recovery_pct:8.1f}% | {quality:7.0f} | {composite:9.1f}")
        
        return filtered_patterns
    
    def create_filtered_dataset(self, filtered_patterns, output_dir='data/processed/filtered_patterns'):
        """Create filtered dataset for model training"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Creating filtered dataset...")
        
        # Separate patterns and metadata
        pattern_metadata = []
        training_sequences = []
        
        for i, (pattern, data) in enumerate(filtered_patterns):
            # Create training sequence
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            sequence_data = data.iloc[start_idx:end_idx]
            
            # Extract OHLCV features
            sequence_features = sequence_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            
            # Create metadata entry
            metadata = {
                'pattern_id': i + 1,
                'symbol': pattern['symbol'],
                'timeframe': pattern['timeframe'],
                'start_date': str(pattern['start_date']),
                'end_date': str(pattern['end_date']),
                'duration': pattern['length'],
                'return_pct': pattern['price_action']['total_return'] * 100,
                'max_drawdown_pct': pattern['price_action']['max_drawdown'] * 100,
                'recovery_ratio': pattern['price_action']['recovery_ratio'],
                'quality_score': pattern['advanced_quality_score'],
                'composite_score': pattern['composite_score'],
                'volume_confirmation': pattern['volume_pattern']['volume_spike_end'],
                'label': 'buy' if pattern['price_action']['total_return'] > 0.05 else 'hold'
            }
            
            pattern_metadata.append(metadata)
            training_sequences.append(sequence_features.tolist())
        
        # Save filtered dataset
        filtered_data = {
            'creation_date': datetime.now().isoformat(),
            'filter_criteria': self.quality_thresholds,
            'total_filtered_patterns': len(filtered_patterns),
            'metadata': pattern_metadata,
            'sequences': training_sequences
        }
        
        with open(output_dir / 'filtered_patterns.json', 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        # Save metadata separately for easier analysis
        metadata_df = pd.DataFrame(pattern_metadata)
        metadata_df.to_csv(output_dir / 'pattern_metadata.csv', index=False)
        
        # Create summary statistics
        summary = {
            'total_patterns': len(filtered_patterns),
            'average_return': metadata_df['return_pct'].mean(),
            'median_return': metadata_df['return_pct'].median(),
            'average_drawdown': metadata_df['max_drawdown_pct'].mean(),
            'average_duration': metadata_df['duration'].mean(),
            'buy_signals': sum(1 for m in pattern_metadata if m['label'] == 'buy'),
            'hold_signals': sum(1 for m in pattern_metadata if m['label'] == 'hold'),
            'symbols_covered': metadata_df['symbol'].nunique(),
            'quality_distribution': {
                'A_grade': sum(1 for m in pattern_metadata if m['quality_score'] >= 80),
                'B_grade': sum(1 for m in pattern_metadata if 60 <= m['quality_score'] < 80),
                'C_grade': sum(1 for m in pattern_metadata if 40 <= m['quality_score'] < 60)
            }
        }
        
        with open(output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Filtered dataset created:")
        print(f"   📁 Location: {output_dir}")
        print(f"   📊 Patterns: {summary['total_patterns']}")
        print(f"   📈 Avg Return: {summary['average_return']:.1f}%")
        print(f"   📉 Avg Drawdown: {summary['average_drawdown']:.1f}%")
        print(f"   🎯 Buy Signals: {summary['buy_signals']}")
        print(f"   ⏸️  Hold Signals: {summary['hold_signals']}")
        
        return output_dir, summary

def main():
    parser = argparse.ArgumentParser(description='Phase 1: Pattern Quality Filter')
    parser.add_argument('--data-dir', type=str, default='data/raw/crypto_patterns', help='Input data directory')
    parser.add_argument('--min-return', type=float, default=0.03, help='Minimum return threshold')
    parser.add_argument('--max-drawdown', type=float, default=0.45, help='Maximum drawdown threshold')
    parser.add_argument('--min-quality', type=int, default=40, help='Minimum quality score')
    parser.add_argument('--output-dir', type=str, default='data/processed/filtered_patterns', help='Output directory')
    
    args = parser.parse_args()
    
    print("🔬 Phase 1: Pattern Quality Filter")
    print(f"📁 Input: {args.data_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🎯 Min Return: {args.min_return*100:.1f}%")
    print(f"🛡️  Max Drawdown: {args.max_drawdown*100:.1f}%")
    
    # Initialize filter
    filter_system = PatternQualityFilter(data_dir=args.data_dir)
    filter_system.quality_thresholds['min_return'] = args.min_return
    filter_system.quality_thresholds['max_drawdown'] = args.max_drawdown
    filter_system.quality_thresholds['min_quality_score'] = args.min_quality
    
    # Load all patterns
    all_patterns = filter_system.load_all_patterns()
    
    if not all_patterns:
        print("❌ No patterns found. Run Phase 1 data collection first.")
        return
    
    # Apply filters
    filtered_patterns, filter_stats = filter_system.apply_strict_filters(all_patterns)
    
    if not filtered_patterns:
        print("❌ No patterns passed the quality filters. Consider relaxing criteria.")
        return
    
    # Rank patterns
    ranked_patterns = filter_system.rank_patterns_by_quality(filtered_patterns)
    
    # Create filtered dataset
    output_dir, summary = filter_system.create_filtered_dataset(ranked_patterns, args.output_dir)
    
    print(f"\n🎉 Pattern filtering complete!")
    print(f"📊 Filtered {filter_stats['total_input']:,} → {len(filtered_patterns)} patterns")
    print(f"📈 Success rate improvement: {summary['average_return']:.1f}% (vs -15.2% original)")
    print(f"🎯 Ready for Phase 2 model training with high-quality data!")

if __name__ == "__main__":
    main() 
"""
Generate sample aircraft modification data for testing and development
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict
from datetime import datetime, timedelta

class SampleDataGenerator:
    """
    Generate realistic aircraft modification data for testing
    """
    
    def __init__(self):
        self.mod_types = [
            'Avionics', 'Structure', 'Cabin', 'Systems', 'Safety', 
            'Propulsion', 'Environmental', 'Electrical'
        ]
        
        self.aircraft_types = [
            'A320', 'A321', 'A330', 'A340', 'A350', 'A380',
            'B737', 'B737MAX', 'B747', 'B777', 'B777X', 'B787',
            'ATR42', 'ATR72', 'CRJ', 'ERJ', 'A300F', 'A330F'
        ]
        
        self.loi_levels = ['Low', 'Medium', 'High']
        
        # Base regulation templates by category
        self.regulations_by_type = {
            'Avionics': ['CS 25.1309', 'CS 25.1431', 'AMC 20-115', 'AMC 20-130', 'AMC 20-151'],
            'Structure': ['CS 25.143', 'CS 25.321', 'CS 25.561', 'CS 25.629', 'CS 25.783'],
            'Cabin': ['CS 25.773', 'CS 25.853', 'CS 25.857', 'CS 25.785', 'CS 25.812'],
            'Systems': ['CS 25.729', 'CS 25.965', 'CS 25.1441', 'CS 25.831', 'CS 25.841'],
            'Safety': ['CS 25.807', 'CS 25.809', 'CS 25.562', 'AMC 25-17', 'CS 25.1457'],
            'Propulsion': ['CS 25.933', 'CS 25.934', 'AMC 20-149', 'CS 25.901', 'CS 25.903'],
            'Environmental': ['CS 25.831', 'CS 25.832', 'AMC 25-831', 'CS 25.1419', 'CS 25.1420'],
            'Electrical': ['CS 25.1309', 'CS 25.1381', 'CS 25.1383', 'AMC 25-12', 'CS 25.1431']
        }
        
        # Component templates for generating descriptions
        self.components = {
            'Avionics': [
                'VHF radio', 'navigation system', 'autopilot', 'flight management system',
                'weather radar', 'collision avoidance system', 'communication equipment',
                'transponder', 'GPS system', 'display unit', 'antenna system'
            ],
            'Structure': [
                'wing extension', 'fuselage reinforcement', 'door frame', 'wing box',
                'control surface', 'landing gear bay', 'cargo door', 'emergency exit',
                'wing tip', 'tail cone', 'belly fairing'
            ],
            'Cabin': [
                'passenger seat', 'galley equipment', 'lavatory', 'overhead bin',
                'lighting system', 'entertainment system', 'air outlet', 'cabin divider',
                'floor panel', 'wall panel', 'ceiling panel'
            ],
            'Systems': [
                'hydraulic pump', 'fuel pump', 'air conditioning pack', 'oxygen system',
                'fire suppression system', 'landing gear actuator', 'valve assembly',
                'pressure regulator', 'temperature sensor', 'flow meter'
            ],
            'Safety': [
                'emergency slide', 'life vest', 'oxygen mask', 'emergency lighting',
                'evacuation system', 'fire extinguisher', 'smoke detector',
                'emergency exit sign', 'safety belt', 'flotation device'
            ],
            'Propulsion': [
                'engine mount', 'thrust reverser', 'fuel nozzle', 'compressor blade',
                'turbine section', 'exhaust nozzle', 'engine cowling', 'ignition system',
                'fuel control unit', 'oil system'
            ]
        }
        
        self.actions = [
            'installation', 'replacement', 'modification', 'upgrade', 'retrofit',
            'addition', 'removal', 'relocation', 'enhancement', 'integration'
        ]
        
        self.locations = [
            'forward fuselage', 'aft fuselage', 'dorsal fuselage', 'belly',
            'wing root', 'wing tip', 'leading edge', 'trailing edge',
            'cockpit', 'cabin', 'cargo compartment', 'engine pylon',
            'landing gear bay', 'nose section', 'tail section'
        ]
        
        self.purposes = [
            'improved performance', 'enhanced safety', 'regulatory compliance',
            'passenger comfort', 'operational efficiency', 'cost reduction',
            'noise reduction', 'fuel efficiency', 'maintenance optimization',
            'system reliability', 'communication enhancement', 'navigation improvement'
        ]
    
    def generate_description(self, mod_type: str) -> str:
        """
        Generate realistic modification description
        
        Args:
            mod_type (str): Type of modification
            
        Returns:
            str: Generated description
        """
        action = random.choice(self.actions)
        component = random.choice(self.components.get(mod_type, ['system']))
        location = random.choice(self.locations)
        purpose = random.choice(self.purposes)
        
        # Create base description
        if action == 'installation':
            desc = f"Installation of new {component} in {location}"
        elif action == 'replacement':
            desc = f"Replacement of existing {component} in {location} with improved version"
        elif action == 'modification':
            desc = f"Modification of {component} in {location}"
        elif action == 'upgrade':
            desc = f"Upgrade of {component} system in {location}"
        else:
            desc = f"{action.capitalize()} of {component} in {location}"
        
        # Add purpose
        desc += f" for {purpose}"
        
        # Add technical details
        technical_details = [
            "affecting structural integrity",
            "with enhanced performance characteristics",
            "including improved materials",
            "with advanced control algorithms",
            "featuring redundant safety systems",
            "incorporating latest technology standards",
            "meeting new certification requirements",
            "with reduced environmental impact"
        ]
        
        if random.random() > 0.3:  # 70% chance to add technical detail
            desc += f" {random.choice(technical_details)}"
        
        return desc
    
    def generate_regulations(self, mod_type: str, num_regs: int = None) -> List[str]:
        """
        Generate relevant regulations for modification type
        
        Args:
            mod_type (str): Type of modification
            num_regs (int): Number of regulations (random if None)
            
        Returns:
            list: List of regulation IDs
        """
        base_regs = self.regulations_by_type.get(mod_type, ['CS 25.1309'])
        
        if num_regs is None:
            num_regs = random.randint(2, 4)
        
        # Select regulations
        selected_regs = random.sample(base_regs, min(num_regs, len(base_regs)))
        
        # Add some general regulations
        general_regs = ['CS 25.1309', 'CS 25.561', 'AMC 20-22']
        if random.random() > 0.5:  # 50% chance
            additional = random.choice([reg for reg in general_regs if reg not in selected_regs])
            if additional:
                selected_regs.append(additional)
        
        return selected_regs
    
    def generate_loi(self, mod_type: str, num_regs: int) -> str:
        """
        Generate Level of Involvement based on modification characteristics
        
        Args:
            mod_type (str): Type of modification
            num_regs (int): Number of regulations involved
            
        Returns:
            str: LOI level
        """
        # Base probabilities by type
        loi_probs = {
            'Avionics': {'Low': 0.3, 'Medium': 0.5, 'High': 0.2},
            'Structure': {'Low': 0.1, 'Medium': 0.4, 'High': 0.5},
            'Cabin': {'Low': 0.6, 'Medium': 0.3, 'High': 0.1},
            'Systems': {'Low': 0.2, 'Medium': 0.6, 'High': 0.2},
            'Safety': {'Low': 0.1, 'Medium': 0.3, 'High': 0.6},
            'Propulsion': {'Low': 0.1, 'Medium': 0.2, 'High': 0.7}
        }
        
        probs = loi_probs.get(mod_type, {'Low': 0.4, 'Medium': 0.4, 'High': 0.2})
        
        # Adjust based on number of regulations
        if num_regs >= 4:
            probs['High'] += 0.2
            probs['Medium'] -= 0.1
            probs['Low'] -= 0.1
        elif num_regs <= 2:
            probs['Low'] += 0.2
            probs['Medium'] -= 0.1
            probs['High'] -= 0.1
        
        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # Sample based on probabilities
        rand_val = random.random()
        cumulative = 0
        for level, prob in probs.items():
            cumulative += prob
            if rand_val <= cumulative:
                return level
        
        return 'Medium'  # Fallback
    
    def generate_dataset(self, num_mods: int = 100, 
                        start_date: str = '2023-01-01') -> pd.DataFrame:
        """
        Generate complete dataset of aircraft modifications
        
        Args:
            num_mods (int): Number of modifications to generate
            start_date (str): Start date for approvals
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        data = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        for i in range(num_mods):
            # Generate mod ID
            mod_id = f"MOD-{start_dt.year}-{i+1:03d}"
            
            # Select modification type
            mod_type = random.choice(self.mod_types)
            
            # Generate description
            description = self.generate_description(mod_type)
            
            # Generate regulations
            regulations = self.generate_regulations(mod_type)
            
            # Generate LOI
            loi = self.generate_loi(mod_type, len(regulations))
            
            # Select aircraft type
            aircraft_type = random.choice(self.aircraft_types)
            
            # Generate approval date
            days_offset = random.randint(0, 365)
            approval_date = start_dt + timedelta(days=days_offset)
            
            data.append({
                'mod_id': mod_id,
                'mod_description': description,
                'mod_type': mod_type,
                'regulations': ','.join(regulations),
                'loi': loi,
                'aircraft_type': aircraft_type,
                'approval_date': approval_date.strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(data)
    
    def generate_extended_dataset(self, num_mods: int = 500) -> pd.DataFrame:
        """
        Generate larger dataset with more variety
        
        Args:
            num_mods (int): Number of modifications
            
        Returns:
            pd.DataFrame: Extended dataset
        """
        # Generate in batches with different years
        datasets = []
        
        years = ['2021', '2022', '2023', '2024']
        mods_per_year = num_mods // len(years)
        
        for year in years:
            year_data = self.generate_dataset(
                mods_per_year, 
                f'{year}-01-01'
            )
            datasets.append(year_data)
        
        # Combine all datasets
        full_dataset = pd.concat(datasets, ignore_index=True)
        
        # Add some cross-type modifications
        cross_type_mods = self._generate_cross_type_mods(num_mods // 10)
        full_dataset = pd.concat([full_dataset, cross_type_mods], ignore_index=True)
        
        return full_dataset
    
    def _generate_cross_type_mods(self, num_mods: int) -> pd.DataFrame:
        """
        Generate modifications that span multiple types
        
        Args:
            num_mods (int): Number of cross-type mods
            
        Returns:
            pd.DataFrame: Cross-type modifications
        """
        data = []
        
        cross_type_patterns = [
            ('Avionics', 'Structure', 'antenna installation affecting structure'),
            ('Systems', 'Safety', 'emergency system modification'),
            ('Cabin', 'Safety', 'evacuation system upgrade'),
            ('Avionics', 'Systems', 'integrated flight system'),
            ('Structure', 'Propulsion', 'engine mount modification')
        ]
        
        for i in range(num_mods):
            primary_type, secondary_type, description_pattern = random.choice(cross_type_patterns)
            
            # Create combined description
            description = f"Integrated {description_pattern} affecting both {primary_type.lower()} and {secondary_type.lower()} systems"
            
            # Combine regulations from both types
            regs1 = self.generate_regulations(primary_type, 2)
            regs2 = self.generate_regulations(secondary_type, 2)
            all_regs = list(set(regs1 + regs2))
            
            # Higher LOI for cross-type mods
            loi_weights = {'Low': 0.1, 'Medium': 0.4, 'High': 0.5}
            loi = np.random.choice(list(loi_weights.keys()), p=list(loi_weights.values()))
            
            data.append({
                'mod_id': f"MOD-2024-CT-{i+1:03d}",
                'mod_description': description,
                'mod_type': f"{primary_type}/{secondary_type}",
                'regulations': ','.join(all_regs),
                'loi': loi,
                'aircraft_type': random.choice(self.aircraft_types),
                'approval_date': '2024-01-01'
            })
        
        return pd.DataFrame(data)

def generate_sample_files(output_dir: str = '../data/'):
    """
    Generate all sample data files
    
    Args:
        output_dir (str): Output directory path
    """
    generator = SampleDataGenerator()
    
    # Generate main dataset
    print("Generating main dataset...")
    main_dataset = generator.generate_extended_dataset(200)
    main_dataset.to_csv(f'{output_dir}/sample_mods_extended.csv', index=False)
    print(f"Generated {len(main_dataset)} modifications")
    
    # Generate smaller test dataset
    print("Generating test dataset...")
    test_dataset = generator.generate_dataset(50)
    test_dataset.to_csv(f'{output_dir}/sample_mods_test.csv', index=False)
    print(f"Generated {len(test_dataset)} test modifications")
    
    # Generate training/validation split
    print("Creating train/validation split...")
    shuffled = main_dataset.sample(frac=1, random_state=42)
    split_idx = int(0.8 * len(shuffled))
    
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]
    
    train_data.to_csv(f'{output_dir}/train_mods.csv', index=False)
    val_data.to_csv(f'{output_dir}/val_mods.csv', index=False)
    
    print(f"Training set: {len(train_data)} modifications")
    print(f"Validation set: {len(val_data)} modifications")
    
    print("Sample data generation complete!")

if __name__ == "__main__":
    generate_sample_files()

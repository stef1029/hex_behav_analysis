#!/usr/bin/env python3
"""
Batch script to fix Axona .set files based on mouse ID configurations.

This script uses the Cohort_folder class to scan sessions and automatically
updates collectMask settings in .set files based on the specified mouse
channel configurations.

Mouse configurations:
- 16 channels (4 tetrodes): mtaq13-3a
- 32 channels (8 tetrodes): mtaq11-3b, mtaq14-1j, mtaq14-1i
"""

import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import json
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder


class AxonaBatchFixer:
    """
    Class to batch fix Axona .set files based on mouse channel configurations.
    """
    
    def __init__(self, cohort_folder_instance):
        """
        Initialise the batch fixer with a Cohort_folder instance.
        
        Parameters
        ----------
        cohort_folder_instance : Cohort_folder
            Instance of the Cohort_folder class containing session information
        """
        self.cohort = cohort_folder_instance
        
        # Define mouse channel configurations
        self.mouse_configs = {
            # 16 channels (4 tetrodes) - tetrodes 1-4 active
            'mtaq13-3a': {
                'channels': 16,
                'tetrodes': 4,
                'active_tetrodes': list(range(1, 5))
            },
            # 32 channels (8 tetrodes) - tetrodes 1-8 active
            'mtaq11-3b': {
                'channels': 32,
                'tetrodes': 8,
                'active_tetrodes': list(range(1, 9))
            },
            'mtaq14-1j': {
                'channels': 32,
                'tetrodes': 8,
                'active_tetrodes': list(range(1, 9))
            },
            'mtaq14-1i': {
                'channels': 32,
                'tetrodes': 8,
                'active_tetrodes': list(range(1, 9))
            }
        }
    
    def find_set_files_for_sessions(self) -> Dict[str, List[Tuple[str, Path]]]:
        """
        Find .set files associated with sessions in the cohort.
        
        Returns
        -------
        Dict[str, List[Tuple[str, Path]]]
            Dictionary mapping mouse IDs to lists of (session_id, set_file_path) tuples
        """
        set_files_by_mouse = {}
        
        for mouse_id in self.cohort.cohort["mice"]:
            set_files_by_mouse[mouse_id] = []
            
            for session_id in self.cohort.cohort["mice"][mouse_id]["sessions"]:
                session_info = self.cohort.cohort["mice"][mouse_id]["sessions"][session_id]
                session_dir = Path(session_info["directory"])
                
                # Check if ephys_data exists and contains a .set file
                if "ephys_data" in session_info:
                    ephys_files = session_info["ephys_data"]
                    # Check if 'set' key exists and contains a valid path
                    if "set" in ephys_files:
                        set_file_path = Path(ephys_files["set"])
                        if set_file_path.exists():
                            set_files_by_mouse[mouse_id].append((session_id, set_file_path))
                            continue
                        else:
                            print(f"    ⚠️  Set file listed but not found: {set_file_path}")
                
                # Fallback: look for .set files in the session directory or parent
                set_file = self._find_set_file_in_directory(session_dir)
                if set_file:
                    set_files_by_mouse[mouse_id].append((session_id, set_file))
                else:
                    # Also check parent directory (group folder)
                    parent_set_file = self._find_set_file_in_directory(session_dir.parent)
                    if parent_set_file:
                        set_files_by_mouse[mouse_id].append((session_id, parent_set_file))
                    else:
                        print(f"    ⚠️  No .set file found for session {session_id}")
        
        return set_files_by_mouse
    
    def _find_set_file_in_directory(self, directory: Path) -> Path:
        """
        Find a .set file in the specified directory.
        
        Parameters
        ----------
        directory : Path
            Directory to search for .set files
            
        Returns
        -------
        Path or None
            Path to the .set file or None if not found
        """
        if not directory.exists():
            return None
            
        set_files = list(directory.glob("*.set"))
        if set_files:
            return set_files[0]  # Return the first .set file found
        return None
    
    def analyse_current_configurations(self) -> Dict[str, Dict]:
        """
        Analyse current collectMask configurations for all sessions.
        
        Returns
        -------
        Dict[str, Dict]
            Analysis results for each mouse
        """
        set_files_by_mouse = self.find_set_files_for_sessions()
        analysis_results = {}
        
        for mouse_id, set_files in set_files_by_mouse.items():
            analysis_results[mouse_id] = {
                'total_sessions': len(set_files),
                'sessions': [],
                'expected_config': self.mouse_configs.get(mouse_id, 'Unknown mouse ID'),
                'needs_fixing': []
            }
            
            for session_id, set_file_path in set_files:
                current_config = self._analyse_set_file(set_file_path)
                analysis_results[mouse_id]['sessions'].append({
                    'session_id': session_id,
                    'set_file': str(set_file_path),
                    'current_active_tetrodes': current_config['active_tetrodes'],
                    'current_channel_count': current_config['channel_count']
                })
                
                # Check if this session needs fixing
                expected_tetrodes = self.mouse_configs.get(mouse_id, {}).get('active_tetrodes', [])
                if current_config['active_tetrodes'] != expected_tetrodes:
                    analysis_results[mouse_id]['needs_fixing'].append(session_id)
        
        return analysis_results
    
    def _analyse_set_file(self, set_file_path: Path) -> Dict:
        """
        Analyse the current collectMask configuration in a .set file.
        
        Parameters
        ----------
        set_file_path : Path
            Path to the .set file to analyse
            
        Returns
        -------
        Dict
            Current configuration information
        """
        try:
            with open(set_file_path, 'r', encoding='cp1252') as file:
                content = file.read()
            
            active_tetrodes = []
            for match in re.finditer(r'collectMask_(\d+)\s+(\d+)', content):
                tetrode_num = int(match.group(1))
                is_active = int(match.group(2))
                if is_active == 1:
                    active_tetrodes.append(tetrode_num)
            
            return {
                'active_tetrodes': sorted(active_tetrodes),
                'channel_count': len(active_tetrodes) * 4
            }
            
        except Exception as e:
            print(f"Error analysing {set_file_path}: {e}")
            return {'active_tetrodes': [], 'channel_count': 0}
    
    def fix_set_file(self, set_file_path: Path, target_tetrodes: List[int], 
                     create_backup: bool = True) -> bool:
        """
        Fix a single .set file by setting the correct collectMask values.
        
        Parameters
        ----------
        set_file_path : Path
            Path to the .set file to fix
        target_tetrodes : List[int]
            List of tetrode numbers that should be active
        create_backup : bool, default True
            Whether to create a backup of the original file
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Read the original file
            with open(set_file_path, 'r', encoding='cp1252') as file:
                content = file.read()
            
            # Create backup if requested
            if create_backup:
                backup_path = set_file_path.with_suffix('.set.backup')
                shutil.copy2(set_file_path, backup_path)
                print(f"    ✓ Backup created: {backup_path.name}")
            
            # Find all collectMask entries and update them
            modified_content = content
            changes_made = []
            
            # First, set all collectMask entries to 0
            for match in re.finditer(r'collectMask_(\d+)\s+\d+', content):
                tetrode_num = int(match.group(1))
                pattern = f'collectMask_{tetrode_num}\\s+\\d+'
                replacement = f'collectMask_{tetrode_num} 0'
                modified_content = re.sub(pattern, replacement, modified_content)
            
            # Then, set target tetrodes to 1
            for tetrode_num in target_tetrodes:
                pattern = f'collectMask_{tetrode_num}\\s+0'
                replacement = f'collectMask_{tetrode_num} 1'
                
                if re.search(pattern, modified_content):
                    modified_content = re.sub(pattern, replacement, modified_content)
                    changes_made.append(tetrode_num)
            
            # Write the modified file
            with open(set_file_path, 'w', encoding='cp1252') as file:
                file.write(modified_content)
            
            print(f"    ✓ Updated tetrodes {target_tetrodes} to active")
            return True
            
        except Exception as e:
            print(f"    ✗ Error fixing {set_file_path}: {e}")
            return False
    
    def fix_all_sessions(self, dry_run: bool = False, create_backups: bool = True) -> None:
        """
        Fix all .set files for all sessions based on mouse configurations.
        
        Parameters
        ----------
        dry_run : bool, default False
            If True, only show what would be changed without making modifications
        create_backups : bool, default True
            Whether to create backup files before modifying
        """
        print("🔧 Axona .set File Batch Fixer")
        print("=" * 50)
        
        # Get current configurations
        set_files_by_mouse = self.find_set_files_for_sessions()
        
        if not set_files_by_mouse:
            print("❌ No .set files found in any sessions")
            return
        
        total_files_processed = 0
        total_files_fixed = 0
        
        for mouse_id, set_files in set_files_by_mouse.items():
            if not set_files:
                continue
                
            print(f"\n🐭 Mouse: {mouse_id}")
            
            # Check if we have a configuration for this mouse
            if mouse_id not in self.mouse_configs:
                print(f"    ⚠️  Unknown mouse ID - skipping")
                continue
            
            config = self.mouse_configs[mouse_id]
            target_tetrodes = config['active_tetrodes']
            target_channels = config['channels']
            
            print(f"    📊 Target configuration: {target_channels} channels ({len(target_tetrodes)} tetrodes)")
            print(f"    🎯 Target active tetrodes: {target_tetrodes}")
            print(f"    📁 Sessions found: {len(set_files)}")
            
            for session_id, set_file_path in set_files:
                total_files_processed += 1
                
                # Analyse current configuration
                current_config = self._analyse_set_file(set_file_path)
                current_tetrodes = current_config['active_tetrodes']
                current_channels = current_config['channel_count']
                
                print(f"\n    📄 Session: {session_id}")
                print(f"         File: {set_file_path.name}")
                print(f"         Current: {current_channels} channels (tetrodes {current_tetrodes})")
                
                # Check if fixing is needed
                if current_tetrodes == target_tetrodes:
                    print(f"         ✅ Already correctly configured")
                    continue
                
                print(f"         🔄 Needs update: {current_channels} → {target_channels} channels")
                
                if dry_run:
                    print(f"         🔍 DRY RUN: Would update tetrodes {current_tetrodes} → {target_tetrodes}")
                else:
                    # Apply the fix
                    success = self.fix_set_file(set_file_path, target_tetrodes, create_backups)
                    if success:
                        total_files_fixed += 1
                        print(f"         ✅ Successfully updated!")
                    else:
                        print(f"         ❌ Failed to update")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Files processed: {total_files_processed}")
        if not dry_run:
            print(f"   Files successfully fixed: {total_files_fixed}")
            print(f"   Files that needed no changes: {total_files_processed - total_files_fixed}")
        else:
            print(f"   Files that would be changed: {total_files_fixed}")
        
        if not dry_run and total_files_fixed > 0:
            print(f"\n✅ Batch fixing complete!")
            print(f"   SpikeInterface should now load the correct number of channels for each mouse.")
            if create_backups:
                print(f"   Original files backed up with .backup extension.")
    
    def generate_report(self) -> None:
        """
        Generate a detailed report of current configurations and required changes.
        """
        print("📊 Axona Configuration Analysis Report")
        print("=" * 60)
        
        analysis = self.analyse_current_configurations()
        
        for mouse_id, results in analysis.items():
            print(f"\n🐭 Mouse: {mouse_id}")
            
            if results['expected_config'] == 'Unknown mouse ID':
                print(f"    ⚠️  Unknown mouse ID - no configuration defined")
                continue
            
            expected = results['expected_config']
            print(f"    📊 Expected: {expected['channels']} channels ({expected['tetrodes']} tetrodes)")
            print(f"    🎯 Expected active tetrodes: {expected['active_tetrodes']}")
            print(f"    📁 Total sessions: {results['total_sessions']}")
            
            if results['needs_fixing']:
                print(f"    🔧 Sessions needing fixes: {len(results['needs_fixing'])}")
                for session_id in results['needs_fixing'][:3]:  # Show first 3
                    print(f"         - {session_id}")
                if len(results['needs_fixing']) > 3:
                    print(f"         ... and {len(results['needs_fixing']) - 3} more")
            else:
                print(f"    ✅ All sessions correctly configured")
            
            # Show detailed breakdown for first few sessions
            print(f"    📋 Sample sessions:")
            for session_info in results['sessions'][:3]:
                status = "✅" if session_info['session_id'] not in results['needs_fixing'] else "🔧"
                print(f"         {status} {session_info['session_id']}: "
                      f"{session_info['current_channel_count']} channels "
                      f"(tetrodes {session_info['current_active_tetrodes']})")


def main():
    """
    Main function to run the batch fixer.
    """
    # You'll need to modify this path to point to your cohort directory
    cohort_directory = Path(r"Z://Behaviour/2504_pitx_ephys_cohort")
    
    # For interactive use, ask user for the cohort directory
    if not cohort_directory.exists():
        user_input = input("Enter path to your cohort directory: ").strip().strip('"')
        cohort_directory = Path(user_input)
    
    if not cohort_directory.exists():
        print(f"❌ Directory not found: {cohort_directory}")
        return
    
    print(f"🔍 Loading cohort information from: {cohort_directory}")

    
    try:
        # Create Cohort_folder instance with ephys data scanning enabled
        cohort = Cohort_folder(
            cohort_directory,
            multi=True,
            portable_data=False,  # Assuming raw data since we're looking for .set files
            use_existing_cohort_info=False,
            ephys_data=True  # Important: enable ephys data scanning
        )
        
        # Create the batch fixer
        fixer = AxonaBatchFixer(cohort)
        
        # Generate analysis report
        fixer.generate_report()
        
        # Ask user what to do
        print(f"\n❓ What would you like to do?")
        print(f"   1. Dry run (show what would be changed)")
        print(f"   2. Fix all files (with backups)")
        print(f"   3. Fix all files (without backups)")
        print(f"   4. Exit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            fixer.fix_all_sessions(dry_run=True)
        elif choice == "2":
            fixer.fix_all_sessions(dry_run=False, create_backups=True)
        elif choice == "3":
            fixer.fix_all_sessions(dry_run=False, create_backups=False)
        elif choice == "4":
            print("👋 Exiting without changes")
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
from pathlib import Path

class SessionScanner:
    def __init__(self, cohort_directory, multi=True, portable_data=False, OEAB_legacy=True):
        """
        Initialize scanner with cohort configuration parameters.
        
        Args:
            cohort_directory (str or Path): Path to the cohort directory
            multi (bool): Whether the cohort uses multi-session structure
            portable_data (bool): Whether using portable data structure
            OEAB_legacy (bool): Whether using legacy OEAB format
        """
        self.cohort_directory = Path(cohort_directory)
        self.multi = multi
        self.portable_data = portable_data
        self.OEAB_legacy = OEAB_legacy
        
        # Default file types to check - can be modified by user
        self.default_file_checks = {
            'behaviour_data': 'behaviour_data',
            'tracker_data': 'Tracker_data',
            'raw_video': '.avi',
            'video_frametimes': 'video_frame_times',
            'sendkey_logs': 'sendkey_logs',
            'nwb_file': '.nwb'
        }
        
        # Add OEAB-specific checks based on legacy setting
        if self.OEAB_legacy:
            self.default_file_checks['arduino_DAQ'] = 'ArduinoDAQ.json'
        else:
            self.default_file_checks['arduino_DAQ'] = 'ArduinoDAQ.h5'

    def find_file(self, directory, tag):
        """Find a file containing the specified tag in its name."""
        for file in directory.glob('*'):
            if tag in file.name:
                return file
        return None

    def get_session_folders(self):
        """Get list of session folders based on cohort structure."""
        if not self.multi:
            return [
                folder for folder in self.cohort_directory.glob('*')
                if len(folder.name) > 13 and folder.name[13] == "_" 
                and folder.is_dir() and 'OEAB_recording' not in folder.name
            ]
        else:
            multi_folders = [
                folder for folder in self.cohort_directory.glob('*')
                if folder.is_dir() and 'OEAB_recording' not in folder.name
            ]
            
            return [
                subfolder for folder in multi_folders 
                for subfolder in folder.glob('*')
                if subfolder.is_dir() and len(subfolder.name) > 13 
                and subfolder.name[13] == "_" and 'OEAB_recording' not in subfolder.name
            ]

    def scan_sessions(self, file_checks=None):
        """
        Scan sessions for specified files.
        
        Args:
            file_checks (dict): Dictionary of file types to check for, format:
                              {name: search_string}
                              If None, uses default checks
        """
        if file_checks is None:
            file_checks = self.default_file_checks
            
        session_folders = self.get_session_folders()
        session_folders.sort(key=lambda x: (x.name[14:], x.name[:13]))
        
        # Initialize statistics
        stats = {
            'total_sessions': 0,
            'files_missing': {key: 0 for key in file_checks},
            'current_mouse': None
        }
        
        print("\nSession File Check Report:")
        print("========================")
        print(f"\nChecking for files: {list(file_checks.keys())}")
        
        for session_folder in session_folders:
            mouse_id = session_folder.name[14:]
            session_id = session_folder.name
            
            # Print mouse header when switching to a new mouse
            if mouse_id != stats['current_mouse']:
                if stats['current_mouse'] is not None:
                    print()  # Add blank line between mice
                stats['current_mouse'] = mouse_id
                print(f"\nMouse ID: {mouse_id}")
            
            print(f"\n  Session: {session_id}")
            print(f"    Path: {session_folder}")
            
            # Check each file type
            for file_type, search_string in file_checks.items():
                file_found = self.find_file(session_folder, search_string)
                status = str(file_found) if file_found else "Not found"
                print(f"    {file_type}: {status}")
                
                if not file_found:
                    stats['files_missing'][file_type] += 1
            
            stats['total_sessions'] += 1
        
        # Print summary
        print("\nSummary:")
        print(f"Total sessions checked: {stats['total_sessions']}")
        print("\nMissing files per type:")
        for file_type, missing_count in stats['files_missing'].items():
            completion_pct = ((stats['total_sessions'] - missing_count) / stats['total_sessions'] * 100)
            print(f"{file_type}:")
            print(f"  Missing: {missing_count}")
            print(f"  Completion: {completion_pct:.1f}%")

def main():
    # Example usage
    cohort_dir = Path("D:/Data/Dan_December_cohort")
    
    # Initialize scanner with cohort configuration
    scanner = SessionScanner(
        cohort_dir,
        multi=True,
        portable_data=False,
        OEAB_legacy=False
    )
    
    # Example of custom file checks
    custom_checks = {
        'behaviour_data': 'behaviour_data',
        'tracker_data': 'Tracker_data',
        'dlc_data': '800000.csv'  # Example of checking for DLC output
    }
    
    # Run scan with custom checks (or use scanner.scan_sessions() for defaults)
    scanner.scan_sessions(custom_checks)

if __name__ == "__main__":
    main()
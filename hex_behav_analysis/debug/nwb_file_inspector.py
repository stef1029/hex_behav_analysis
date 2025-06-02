#!/usr/bin/env python3
"""
NWB File Inspector Script

Inspects NWB files to identify structural differences and dimension mismatches.
Useful for debugging problematic NWB files.

Usage:
    python nwb_inspector.py /path/to/nwb/file.nwb
    python nwb_inspector.py /path/to/directory/with/nwb/files/
"""

import warnings
from pathlib import Path
import sys
import json
from pynwb import NWBHDF5IO
import numpy as np


def capture_warnings(func, *args, **kwargs):
    """Capture warnings during function execution"""
    captured_warnings = []
    
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        captured_warnings.append({
            'message': str(message),
            'category': category.__name__,
            'filename': filename,
            'lineno': lineno
        })
    
    old_showwarning = warnings.showwarning
    warnings.showwarning = warning_handler
    
    try:
        result = func(*args, **kwargs)
        return result, captured_warnings
    finally:
        warnings.showwarning = old_showwarning


def inspect_nwb_file(nwb_path, verbose=True):
    """
    Inspect a single NWB file and return detailed information about its structure.
    
    :param nwb_path: Path to the NWB file
    :param verbose: Whether to print detailed information
    :return: Dictionary containing file information
    """
    nwb_path = Path(nwb_path)
    if verbose:
        print(f"\n{'='*80}")
        print(f"INSPECTING: {nwb_path.name}")
        print(f"Full path: {nwb_path}")
        print(f"{'='*80}")
    
    file_info = {
        'file_path': str(nwb_path),
        'file_name': nwb_path.name,
        'file_exists': nwb_path.exists(),
        'file_size_mb': None,
        'warnings': [],
        'error': None,
        'session_info': {},
        'acquisition': {},
        'processing_modules': {},
        'stimulus': {},
        'dimension_issues': []
    }
    
    if not nwb_path.exists():
        file_info['error'] = f"File does not exist: {nwb_path}"
        if verbose:
            print(f"ERROR: File does not exist")
        return file_info
    
    file_info['file_size_mb'] = round(nwb_path.stat().st_size / (1024 * 1024), 2)
    
    try:
        # Open NWB file with warning capture
        def read_nwb():
            with NWBHDF5IO(str(nwb_path), 'r') as io:
                return io.read()
        
        nwbfile, captured_warnings = capture_warnings(read_nwb)
        file_info['warnings'] = captured_warnings
        
        if verbose and captured_warnings:
            print(f"\nWARNINGS DETECTED ({len(captured_warnings)}):")
            for i, warning in enumerate(captured_warnings, 1):
                print(f"  {i}. {warning['category']}: {warning['message']}")
        
        # Basic file information
        file_info['session_info'] = {
            'session_description': nwbfile.session_description,
            'session_start_time': str(nwbfile.session_start_time),
            'identifier': nwbfile.identifier,
            'experiment_description': nwbfile.experiment_description,
            'experimenter': nwbfile.experimenter,
            'institution': nwbfile.institution,
            'lab': nwbfile.lab
        }
        
        if verbose:
            print(f"\nSESSION INFO:")
            print(f"  Description: {nwbfile.session_description}")
            print(f"  Start time: {nwbfile.session_start_time}")
            print(f"  Experiment: {nwbfile.experiment_description}")
            print(f"  File size: {file_info['file_size_mb']} MB")
        
        # Inspect acquisition data
        if verbose:
            print(f"\nACQUISITION DATA:")
        
        for name, obj in nwbfile.acquisition.items():
            acq_info = inspect_timeseries_object(obj, name, verbose)
            file_info['acquisition'][name] = acq_info
        
        # Inspect processing modules
        if verbose:
            print(f"\nPROCESSING MODULES:")
        
        if hasattr(nwbfile, 'processing'):
            for module_name, module in nwbfile.processing.items():
                if verbose:
                    print(f"  Module: {module_name}")
                    print(f"    Description: {module.description}")
                
                module_info = {
                    'description': module.description,
                    'data_interfaces': {}
                }
                
                if hasattr(module, 'data_interfaces'):
                    for interface_name, interface in module.data_interfaces.items():
                        if hasattr(interface, 'time_series'):
                            # This is a container with TimeSeries
                            if verbose:
                                print(f"    Interface: {interface_name} (container)")
                            interface_info = {'time_series': {}}
                            for ts_name, ts_obj in interface.time_series.items():
                                ts_info = inspect_timeseries_object(ts_obj, ts_name, verbose, indent="      ")
                                interface_info['time_series'][ts_name] = ts_info
                            module_info['data_interfaces'][interface_name] = interface_info
                        else:
                            # This is a direct TimeSeries or other data interface
                            ts_info = inspect_timeseries_object(interface, interface_name, verbose, indent="    ")
                            module_info['data_interfaces'][interface_name] = ts_info
                
                file_info['processing_modules'][module_name] = module_info
        else:
            if verbose:
                print("  No processing modules found")
        
        # Inspect stimulus data
        if verbose:
            print(f"\nSTIMULUS DATA:")
        
        if hasattr(nwbfile, 'stimulus'):
            for name, obj in nwbfile.stimulus.items():
                stim_info = inspect_timeseries_object(obj, name, verbose)
                file_info['stimulus'][name] = stim_info
        else:
            if verbose:
                print("  No stimulus data found")
        
        # Check for dimension issues
        dimension_issues = find_dimension_issues(file_info)
        file_info['dimension_issues'] = dimension_issues
        
        if verbose and dimension_issues:
            print(f"\nDIMENSION ISSUES FOUND:")
            for issue in dimension_issues:
                print(f"  {issue}")
        
    except Exception as e:
        file_info['error'] = str(e)
        if verbose:
            print(f"ERROR reading NWB file: {e}")
            import traceback
            traceback.print_exc()
    
    if verbose:
        print(f"{'='*80}\n")
    
    return file_info


def inspect_timeseries_object(obj, name, verbose=True, indent="  "):
    """
    Inspect a TimeSeries-like object and return information about its dimensions.
    """
    obj_info = {
        'name': name,
        'type': type(obj).__name__,
        'description': getattr(obj, 'description', 'No description'),
        'unit': getattr(obj, 'unit', 'No unit'),
        'data_shape': None,
        'timestamps_shape': None,
        'data_dtype': None,
        'timestamps_dtype': None,
        'dimension_match': None,
        'has_data': False,
        'has_timestamps': False,
        'sample_data': None,
        'sample_timestamps': None
    }
    
    try:
        # Check for data
        if hasattr(obj, 'data') and obj.data is not None:
            obj_info['has_data'] = True
            try:
                obj_info['data_shape'] = obj.data.shape
                obj_info['data_dtype'] = str(obj.data.dtype)
                # Get a small sample of data (first few values)
                if len(obj.data) > 0:
                    sample_size = min(3, len(obj.data))
                    obj_info['sample_data'] = obj.data[:sample_size].tolist()
            except Exception as e:
                obj_info['data_error'] = str(e)
        
        # Check for timestamps
        if hasattr(obj, 'timestamps') and obj.timestamps is not None:
            obj_info['has_timestamps'] = True
            try:
                obj_info['timestamps_shape'] = obj.timestamps.shape
                obj_info['timestamps_dtype'] = str(obj.timestamps.dtype)
                # Get a small sample of timestamps
                if len(obj.timestamps) > 0:
                    sample_size = min(3, len(obj.timestamps))
                    obj_info['sample_timestamps'] = obj.timestamps[:sample_size].tolist()
            except Exception as e:
                obj_info['timestamps_error'] = str(e)
        
        # Check dimension compatibility
        if obj_info['data_shape'] and obj_info['timestamps_shape']:
            data_len = obj_info['data_shape'][0]
            timestamp_len = obj_info['timestamps_shape'][0]
            obj_info['dimension_match'] = (data_len == timestamp_len)
            
            if verbose:
                match_status = "✓" if obj_info['dimension_match'] else "✗"
                print(f"{indent}{name} ({obj_info['type']}):")
                print(f"{indent}  Data shape: {obj_info['data_shape']}")
                print(f"{indent}  Timestamps shape: {obj_info['timestamps_shape']}")
                print(f"{indent}  Dimension match: {match_status}")
                if not obj_info['dimension_match']:
                    print(f"{indent}  ⚠️  MISMATCH: {data_len} data points vs {timestamp_len} timestamps")
        else:
            if verbose:
                print(f"{indent}{name} ({obj_info['type']}):")
                if not obj_info['has_data']:
                    print(f"{indent}  No data attribute")
                if not obj_info['has_timestamps']:
                    print(f"{indent}  No timestamps attribute")
    
    except Exception as e:
        obj_info['inspection_error'] = str(e)
        if verbose:
            print(f"{indent}{name}: Error during inspection - {e}")
    
    return obj_info


def find_dimension_issues(file_info):
    """Find all dimension mismatches in a file info dictionary"""
    issues = []
    
    # Check acquisition data
    for name, obj_info in file_info['acquisition'].items():
        if obj_info.get('dimension_match') is False:
            issues.append(f"Acquisition '{name}': {obj_info['data_shape']} vs {obj_info['timestamps_shape']}")
    
    # Check processing modules
    for module_name, module_info in file_info['processing_modules'].items():
        for interface_name, interface_info in module_info['data_interfaces'].items():
            if 'time_series' in interface_info:
                # Interface contains TimeSeries
                for ts_name, ts_info in interface_info['time_series'].items():
                    if ts_info.get('dimension_match') is False:
                        issues.append(f"Processing '{module_name}/{interface_name}/{ts_name}': {ts_info['data_shape']} vs {ts_info['timestamps_shape']}")
            else:
                # Interface is a direct TimeSeries
                if interface_info.get('dimension_match') is False:
                    issues.append(f"Processing '{module_name}/{interface_name}': {interface_info['data_shape']} vs {interface_info['timestamps_shape']}")
    
    # Check stimulus data
    for name, obj_info in file_info['stimulus'].items():
        if obj_info.get('dimension_match') is False:
            issues.append(f"Stimulus '{name}': {obj_info['data_shape']} vs {obj_info['timestamps_shape']}")
    
    return issues


def compare_files(file_infos):
    """Compare multiple file info dictionaries and highlight differences"""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Group files by whether they have warnings
    files_with_warnings = []
    files_without_warnings = []
    
    for info in file_infos:
        if info['warnings']:
            files_with_warnings.append(info)
        else:
            files_without_warnings.append(info)
    
    print(f"\nFiles with warnings: {len(files_with_warnings)}")
    for info in files_with_warnings:
        print(f"  {info['file_name']} - {len(info['warnings'])} warnings")
    
    print(f"\nFiles without warnings: {len(files_without_warnings)}")
    for info in files_without_warnings:
        print(f"  {info['file_name']}")
    
    # Compare structure differences
    if files_with_warnings and files_without_warnings:
        print(f"\nSTRUCTURE COMPARISON:")
        
        # Pick one example from each group
        problem_file = files_with_warnings[0]
        normal_file = files_without_warnings[0]
        
        print(f"\nProblem file: {problem_file['file_name']}")
        print(f"Normal file: {normal_file['file_name']}")
        
        # Compare processing modules
        problem_modules = set(problem_file['processing_modules'].keys())
        normal_modules = set(normal_file['processing_modules'].keys())
        
        print(f"\nProcessing modules:")
        print(f"  Problem file: {problem_modules}")
        print(f"  Normal file: {normal_modules}")
        
        if problem_modules != normal_modules:
            print(f"  Difference: {problem_modules.symmetric_difference(normal_modules)}")
        
        # Compare TimeSeries in behaviour_coords if present
        if 'behaviour_coords' in problem_file['processing_modules'] and 'behaviour_coords' in normal_file['processing_modules']:
            problem_ts = set(problem_file['processing_modules']['behaviour_coords']['data_interfaces'].keys())
            normal_ts = set(normal_file['processing_modules']['behaviour_coords']['data_interfaces'].keys())
            
            print(f"\nTimeSeries in behaviour_coords:")
            print(f"  Problem file: {problem_ts}")
            print(f"  Normal file: {normal_ts}")
            
            if problem_ts != normal_ts:
                print(f"  Difference: {problem_ts.symmetric_difference(normal_ts)}")


def main():
    """Main function to handle command line arguments and run inspection"""
    if len(sys.argv) < 2:
        print("Usage: python nwb_inspector.py <nwb_file_or_directory>")
        print("Examples:")
        print("  python nwb_inspector.py file.nwb")
        print("  python nwb_inspector.py /path/to/nwb/files/")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)
    
    # Collect NWB files
    nwb_files = []
    if input_path.is_file() and input_path.suffix == '.nwb':
        nwb_files = [input_path]
    elif input_path.is_dir():
        nwb_files = list(input_path.glob("*.nwb"))
        if not nwb_files:
            print(f"No .nwb files found in directory: {input_path}")
            sys.exit(1)
    else:
        print(f"Error: Input must be a .nwb file or directory containing .nwb files")
        sys.exit(1)
    
    print(f"Found {len(nwb_files)} NWB file(s) to inspect")
    
    # Inspect each file
    file_infos = []
    for nwb_file in nwb_files:
        file_info = inspect_nwb_file(nwb_file, verbose=True)
        file_infos.append(file_info)
    
    # If multiple files, do comparison
    if len(file_infos) > 1:
        compare_files(file_infos)
    
    # Optionally save results to JSON
    if len(file_infos) > 0:
        output_file = input_path.parent / "nwb_inspection_results.json" if input_path.is_file() else input_path / "nwb_inspection_results.json"
        
        # Convert to JSON-serializable format
        json_data = []
        for info in file_infos:
            # Remove non-serializable items
            json_info = info.copy()
            # Convert numpy arrays in sample data to lists
            def convert_samples(obj_info):
                if 'sample_data' in obj_info and obj_info['sample_data'] is not None:
                    try:
                        obj_info['sample_data'] = [float(x) if isinstance(x, np.number) else x for x in obj_info['sample_data']]
                    except:
                        obj_info['sample_data'] = str(obj_info['sample_data'])
                if 'sample_timestamps' in obj_info and obj_info['sample_timestamps'] is not None:
                    try:
                        obj_info['sample_timestamps'] = [float(x) if isinstance(x, np.number) else x for x in obj_info['sample_timestamps']]
                    except:
                        obj_info['sample_timestamps'] = str(obj_info['sample_timestamps'])
                return obj_info
            
            # Apply to all TimeSeries objects
            for obj_info in json_info['acquisition'].values():
                convert_samples(obj_info)
            
            json_data.append(json_info)
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
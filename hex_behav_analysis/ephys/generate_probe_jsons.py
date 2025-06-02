
#!/usr/bin/env python3
"""
Create tetrode probe configuration files.
Generates two JSON files: one for 4 tetrodes and one for 8 tetrodes.
"""

from probeinterface import ProbeGroup, generate_tetrode, write_probeinterface


def main():
    """
    Create probe configuration files for 4 and 8 tetrode configurations.
    """
    # Configuration for 4 tetrodes (16 channels)
    print("Creating 4-tetrode configuration...")
    probegroup_4 = ProbeGroup()
    
    for i in range(4):
        # Generate a standard tetrode
        probe = generate_tetrode(r=15)  # r is the radius of the tetrode bundle in micrometres
        probe = probe.to_3d(axes='xy')
        
        # Position tetrodes in a line with 250 micrometre spacing
        probe.move([i * 250, 0, 0])
        
        # Add to group
        probegroup_4.add_probe(probe)
    
    # Set channel indices (0-15 for 4 tetrodes)
    probegroup_4.set_global_device_channel_indices(list(range(16)))
    
    # Save the 4-tetrode configuration
    write_probeinterface('tetrodes_4.json', probegroup_4)
    print("✅ Saved: tetrodes_4.json (16 channels)")
    
    
    # Configuration for 8 tetrodes (32 channels)
    print("\nCreating 8-tetrode configuration...")
    probegroup_8 = ProbeGroup()
    
    # Arrange 8 tetrodes in a 2x4 grid
    tetrode_index = 0
    for row in range(2):
        for col in range(4):
            # Generate tetrode
            probe = generate_tetrode(r=15)
            probe = probe.to_3d(axes='xy')
            
            # Position in grid with 250 micrometre spacing
            x_pos = col * 250
            y_pos = row * 250
            probe.move([x_pos, y_pos, 0])
            
            # Add to group
            probegroup_8.add_probe(probe)
            tetrode_index += 1
    
    # Set channel indices (0-31 for 8 tetrodes)
    probegroup_8.set_global_device_channel_indices(list(range(32)))
    
    # Save the 8-tetrode configuration
    write_probeinterface('tetrodes_8.json', probegroup_8)
    print("✅ Saved: tetrodes_8.json (32 channels)")
    
    print("\nDone! Files created:")
    print("  - tetrodes_4.json (4 tetrodes, 16 channels)")
    print("  - tetrodes_8.json (8 tetrodes, 32 channels)")


if __name__ == "__main__":
    main()
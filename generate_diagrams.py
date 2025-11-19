"""
Generate UML Diagrams from UML_DIAGRAMS.md

Extracts PlantUML code blocks and generates PNG images.
"""

import re
import subprocess
from pathlib import Path

def extract_plantuml_blocks(markdown_file):
    """Extract all PlantUML code blocks from markdown."""
    with open(markdown_file, 'r') as f:
        content = f.read()

    # Find all PlantUML blocks
    pattern = r'```plantuml\s*@startuml\s+(\w+)(.*?)@enduml\s*```'
    matches = re.findall(pattern, content, re.DOTALL)

    diagrams = []
    for name, code in matches:
        diagrams.append({
            'name': name,
            'code': f'@startuml {name}\n{code}\n@enduml'
        })

    return diagrams

def generate_diagram(diagram, output_dir, plantuml_jar):
    """Generate PNG from PlantUML code."""
    name = diagram['name']
    code = diagram['code']

    # Write PlantUML file
    puml_file = output_dir / f'{name}.puml'
    with open(puml_file, 'w') as f:
        f.write(code)

    # Generate PNG
    print(f'Generating {name}.png...')
    result = subprocess.run(
        ['java', '-jar', str(plantuml_jar), '-tpng', str(puml_file)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f'  ✓ {name}.png created')
        # Remove .puml file after successful generation
        puml_file.unlink()
        return True
    else:
        print(f'  ✗ Error generating {name}.png')
        print(f'    {result.stderr}')
        return False

def main():
    # Paths
    project_root = Path(__file__).parent
    markdown_file = project_root / 'UML_DIAGRAMS.md'
    output_dir = project_root / 'diagrams'
    plantuml_jar = project_root / 'plantuml.jar'

    # Check files exist
    if not markdown_file.exists():
        print(f'Error: {markdown_file} not found')
        return

    if not plantuml_jar.exists():
        print(f'Error: {plantuml_jar} not found')
        print('Download from: https://plantuml.com/download')
        return

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Extract diagrams
    print('Extracting PlantUML diagrams from UML_DIAGRAMS.md...')
    diagrams = extract_plantuml_blocks(markdown_file)
    print(f'Found {len(diagrams)} diagrams\n')

    # Generate PNGs
    print('Generating PNG images...\n')
    success_count = 0
    for diagram in diagrams:
        if generate_diagram(diagram, output_dir, plantuml_jar):
            success_count += 1

    print(f'\n{"="*70}')
    print(f'✓ Generated {success_count}/{len(diagrams)} diagrams successfully')
    print(f'{"="*70}')
    print(f'Output directory: {output_dir}')

    # List generated files
    print('\nGenerated files:')
    for png_file in sorted(output_dir.glob('*.png')):
        size_kb = png_file.stat().st_size / 1024
        print(f'  - {png_file.name} ({size_kb:.1f} KB)')

if __name__ == '__main__':
    main()

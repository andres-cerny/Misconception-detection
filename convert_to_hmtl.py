from pretty_jupyter import convert

convert(
    'similarity_search.ipynb', 
    'output.html',
    title='My Notebook',  # Custom page title
    theme='light',        # 'light' or 'dark'
    input_hiding=True,    # Option to hide input cells
    output_hiding=False   # Option to hide output cells
)
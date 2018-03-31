import os

if os.environ.get('DISPLAY') == '':
    print('No display found. Using non-interactive matplotlib Agg backend.')
    import matplotlib
    matplotlib.use('Agg')

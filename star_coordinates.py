import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from astroquery.simbad import Simbad
import astropy.units as u
from astropy.coordinates import SkyCoord

app = Flask(__name__)
CORS(app)

# Load HYG Database
hyg_data = pd.read_csv('hygdata_v3.csv')

@app.route('/api/star', methods=['GET'])
def get_star_data():
    star_name = request.args.get('name')
    
    # Check HYG Database
    hyg_result = hyg_data[hyg_data['proper'].str.lower() == star_name.lower()]
    
    if not hyg_result.empty:
        star = hyg_result.iloc[0]
        return jsonify({
            'name': star['proper'],
            'coordinates': {
                'rightAscension': star['ra'],
                'declination': star['dec']
            },
            'type': star['spect'],
            'magnitude': star['mag'],
            'distance': star['dist'],
            'description': f"Star from the HYG Database with ID {star['id']}"
        })
    
    # If not found in HYG, query SIMBAD
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('distance', 'sptype', 'flux(V)')
    result_table = custom_simbad.query_object(star_name)
    
    if result_table is not None:
        star = result_table[0]
        coords = SkyCoord(ra=star['RA'], dec=star['DEC'], unit=(u.hourangle, u.deg))
        
        return jsonify({
            'name': star['MAIN_ID'],
            'coordinates': {
                'rightAscension': coords.ra.hour,
                'declination': coords.dec.degree
            },
            'type': star['SP_TYPE'],
            'magnitude': star['FLUX_V'],
            'distance': star['Distance_distance'],
            'description': f"Star data from SIMBAD database."
        })
    
    return jsonify({'error': 'Star not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)
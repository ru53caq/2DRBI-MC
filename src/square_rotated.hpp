 /* Square lattice with PBC
*/

#pragma once
#include <cmath>
namespace lattice{
class square {
	public:
		square() {}
		square(size_t L): L(L) {}



		int nb_1(int i){		//UP-LEFT
			if ( i%(L+1) < (L+1)/2)
				return (i + (L+1)/2   ) % ((L+1)*(L-1)/2 );
			else
				return (i + (L+1)/2 -1) % ((L+1)*(L-1)/2 );
		}

		int nb_2(int i){		//UP-RIGHT
			if ( i%(L+1) < (L+1)/2)
				return (i + (L+1)/2 +1) % ((L+1)*(L-1)/2 );
			else
				return (i + (L+1)/2   ) % ((L+1)*(L-1)/2 );

		}

		int nb_3(int i){		//DOWN-LEFT
			if (i% (L+1) == (L+1)/2)
				return i-1;			// must couple to something with coupling strength 0

			if ( i%(L+1) < (L+1)/2)
				return (i + (L+1)*(L-2)/2   ) % ((L+1)*(L-1)/2 );
			else
				return (i + (L+1)*(L-2)/2 -1) % ((L+1)*(L-1)/2 );



		}

		int nb_4(int i){		//DOWN-RIGHT
			if (i % (L+1) == (L+1)/2 - 1)
				return i + 1;		// must couple to something with coupling strength 0

			if ( i%(L+1) < (L+1)/2)
				return (i + (L+1)*(L-2)/2 +1) % ((L+1)*(L-1)/2 );
			else
				return (i + (L+1)*(L-2)/2   ) % ((L+1)*(L-1)/2 );

		}

	private:
		size_t L;
};
}

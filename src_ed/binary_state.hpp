#ifndef BinaryState_h
#define BinaryState_h

#include <cstdio>
#include <string>
#include <map>
#include <cstdint>

using namespace std;

uint32_t bitcount32(uint32_t n);
uint64_t bitcount64(uint64_t n);
void BinaryToString(size_t n, char *s, size_t m);
void PrintBinaryDouble(std::ostream &flux, const uint64_t &n, size_t m);
uint64_t collapse(const uint64_t &b, const vector<int> &s);

//! The struct binary_state is used to represent basis states in the Hilbert space
struct binary_state
{
	uint64_t b; // the 64 bit representation (max 32 sites, because of spin)
  	static uint64_t leftmask; // mask for the left part of the state (spin up)
	static uint64_t rightmask; // mask for the left part of the state (spin down)
	
	binary_state(): b(0) {}
	binary_state(uint64_t _b): b(_b) {}
	binary_state(uint32_t L, uint32_t R): b((uint64_t)R+((uint64_t)L<<32)) {}
	binary_state(uint32_t R): b((uint64_t)R){}
	binary_state(const binary_state &bin): b(bin.b) {}
	inline size_t count(){return bitcount64(b);} //!< returns the number of set bits
	inline uint32_t left() const {return (uint32_t)(b >> 32);} //!< returns the left part of the binary representation
	inline uint32_t right() const {return (uint32_t)(b&rightmask);} //!< returns the right part of the binary representation
	inline size_t double_occupancy() {return bitcount32(left()&right());}
	
	/**
	 returns -1 if an odd number of bits are set between positions defined by masks i and j. otherwise returns 1.
	 */
	inline int interphase(uint64_t i, uint64_t j)
	{
		if(i==j) return 1;
		uint64_t mask_inside = (i>j)? (i-(j<<1)) : (j-(i<<1));
		return (bitcount64(mask_inside&b)&1) ? -1 : 1;
	}
	
	int one_body(uint64_t b, uint64_t a);
	int pair_annihilate(uint64_t i, uint64_t j);
	int pair_create(uint64_t i, uint64_t j);

	static void flip_spin(uint64_t &x)
	{
		uint64_t tmp = x&rightmask;
		x >>= 32;
		x += (tmp << 32);
	}
	
	static inline uint64_t	mask(size_t i, size_t _n)
	{
    return (i >= _n) ? (uint64_t(1) << (i-_n+32)) : (uint64_t(1) << i);
	}
};

inline bool operator==(const binary_state &b1, const binary_state &b2){return (b1.b==b2.b);}
inline bool operator<(const binary_state &b1, const binary_state &b2){return (b1.b<b2.b);}
inline bool operator>(const binary_state &b1, const binary_state &b2){return (b1.b>b2.b);}


#endif

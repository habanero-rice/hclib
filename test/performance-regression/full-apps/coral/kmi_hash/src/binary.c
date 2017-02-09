/* Large binary division, add, and subtraction.
 */
#include "kmi.h"

static const unsigned char KMI_mask8[] = {
    0xff,
    0x7f,
    0x3f,
    0x1f,
    0xf,
    0x7,
    0x3,
    0x1
};

static const unsigned char KMI_mask1[] = {
    0x80,
    0x40,
    0x20,
    0x10,
    0x8,
    0x4,
    0x2,
    0x1
};


/** Binary array comparison.
 * If s1 > s2, return 1.
 * If s1 == s2, return 0.
 * If s1 < s2, return -1.
 */
int KMI_binary_cmp(char *s1, char *s2, long len)
{
    int i;
    for (i = 0; i < len; i++) {
        int index = i / 8;
        int bitpos = i % 8;
        int x1 = (int) (s1[index] & KMI_mask1[bitpos]);
        int x2 = (int) (s2[index] & KMI_mask1[bitpos]);
        if (x1 > x2)
            return 1;
        else if (x1 < x2)
            return -1;
    }
    return 0;
}

/** Large binary addition: s1 + s2 => s3.
 * Note s1 should be the same length of s2. Based on practical use, 
 * s1 + s2 will not result in a overflow of length str_len.
 * @input s1
 * @input s2
 * @input str_len
 * @output s3
 */
int KMI_binary_add(char *s1, char *s2, long str_len, char *s3)
{
    int bin_len = KMI_BIN_STR_LEN(str_len);
    memset(s3, 0, bin_len);
    int i, j;
    int carrier = 0;
    for (i = str_len - 1; i >= 0; i--) {
        j = i / 4;
        int bitpos = (i % 4) * 2;
        int x1 = (int) ((s1[j] >> (6 - bitpos)) & 0x3);
        int x2 = (int) ((s2[j] >> (6 - bitpos)) & 0x3);
        int tmp = x1 + x2 + carrier;
        if (tmp <= 3) {
            carrier = 0;
            s3[j] |= (char) (tmp << (6 - bitpos));
        } else {
            tmp -= 4;
            carrier = 1;
            s3[j] |= (char) (tmp << (6 - bitpos));
        }
    }
    if (carrier != 0) {
        printf("[error]s3 overflow, s1: ");
        debug_print_bin_str(s1, str_len);
        printf(" s2: ");
        debug_print_bin_str(s2, str_len);
        printf(" s3: ");
        debug_print_bin_str(s3, str_len);
        printf("\n");
        exit(0);
    }

    return 1;
}

/** Large binary subtraction: s1 - s2 => s3.
 * Note s1 should be the same length of s2, and is always bigger than s2.
 * @input s1
 * @input s2
 * @input str_len
 * @output s3
 */
int KMI_binary_sub(char *s1, char *s2, long str_len, char *s3)
{
    int bin_len = KMI_BIN_STR_LEN(str_len);
    memset(s3, 0, bin_len);
    int i, j;
    int carrier = 0;
    for (i = str_len - 1; i >= 0; i--) {
        j = i / 4;
        int bitpos = (i % 4) * 2;
        int x1 = (int) ((s1[j] >> (6 - bitpos)) & 0x3);
        int x2 = (int) ((s2[j] >> (6 - bitpos)) & 0x3);
        int tmp = x1 - x2 - carrier;
        if (tmp >= 0) {
            carrier = 0;
            s3[j] |= (char) (tmp << (6 - bitpos));
        } else {
            tmp += 4;
            carrier = 1;
            s3[j] |= (char) (tmp << (6 - bitpos));
        }
    }
    if (carrier != 0) {
        printf("[error]s1 is smaller than s2\n");
        exit(0);
    }

    return 1;
}

/** Large binary division: s1 / num => s2.
 * This function is dedicated for KMI use, the *s2 array is of the same
 * length of s1. Note len is the original length of s1.
 * @input s1: dividend
 * @input len
 * @input num
 * @output s2
 */
int KMI_binary_div(char *s1, long len, int num, char *s2)
{
    int num_len = 0;
    int x = num;
    long i;
    int j;
    int t1, t2, t3;
    int blen = KMI_BIN_STR_LEN(len);
    memset(s2, 0, blen);

    while (x) {
        num_len++;
        x = x >> 1;
    }
    assert(num_len < 32);

    t1 = 0;
    int index;
    int bitpos;
    for (j = 0; j < num_len - 1; j++) {
        index = j / 8;
        bitpos = (j % 8);
        if ((s1[index] & KMI_mask1[bitpos]) != 0)
            t1 |= 1;
        t1 = t1 << 1;
    }
    index = j / 8;
    bitpos = (j % 8);
    if ((s1[index] & KMI_mask1[bitpos]) != 0)
        t1 |= 1;

    int s2_index;
    int s2_bitpos;
    for (i = 0; i < len * 2 - num_len; i++) {
        s2_index = (i + num_len - 1) / 8;
        s2_bitpos = (i + num_len - 1) % 8;
        if (t1 >= num) {
            t1 = t1 - num;
            s2[s2_index] |= KMI_mask1[s2_bitpos];
        }
        t1 = t1 << 1;
        index = (i + num_len) / 8;
        bitpos = (i + num_len) % 8;
        if ((s1[index] & KMI_mask1[bitpos]) != 0)
            t1 |= 1;
    }
    s2_index = (i + num_len - 1) / 8;
    s2_bitpos = (i + num_len - 1) % 8;
    if (t1 >= num) {
        t1 = t1 - num;
        s2[s2_index] |= KMI_mask1[s2_bitpos];
    }

}

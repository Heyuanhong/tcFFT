#include <fftw3.h>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "doit.h"
extern char *optarg;
extern int optopt;

void fftw3_get_result(double *data, double *result, int n, int n_batch)
{
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan p = fftw_plan_dft_1d(n, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
    
    for (int i = 0; i < n_batch; ++i)
    {
        memcpy(in, data + 2 * i * n, sizeof(fftw_complex) * n);
        fftw_execute(p);
        memcpy(result + 2 * i * n, in, sizeof(fftw_complex) * n);
    }
    
    fftw_destroy_plan(p);
    fftw_free(in);
}

double AbsError(double a, double b)
{

    return fabs(a - b);
}

double RelativeError(double a, double b)
{
    double max_num = std::max(a, b);

    // if (max_num < 1e-5 )
    //     return 0;

    return fabs(a - b) / max_num;
}


double get_error(double *tested, double *standard, int n, int n_batch, double &max_abs_error, double &max_rel_error)

{
    int correct_num = 0;
    max_abs_error = 0;
    max_rel_error = 0;
    for (int i = 0; i < n * n_batch ; i++)
    {
        double e1 = RelativeError(standard[2 * i + 0], tested[2 * i + 0]);
        double e2 = RelativeError(standard[2 * i + 1], tested[2 * i + 1]);
        double e3 = AbsError(standard[2 * i + 0], tested[2 * i + 0]);
        double e4 = AbsError(standard[2 * i + 1], tested[2 * i + 1]);

        max_abs_error = std::max(max_abs_error, std::max(e3, e4));
        max_rel_error = std::max(max_rel_error, std::max(e1, e2));

        if (e1 < 1e-3 && e2 < 1e-3)
        {
            correct_num++;
        }
    }
    printf("correct_num: %d\n", correct_num);
    return (double)correct_num / ((double)n * (double)n_batch);
}

int CompareEpscInterleave(double *tested, double *standard, int n, int n_batch,  double &real_error, double &imag_error)
{
    int correct_num = 0;
    real_error = 0.0;
    imag_error = 0.0;

    double epsc_error_sum_real = 0.0;
    double epsc_error_sum_imag = 0.0;
    double two_norm_error_sum_real = 0.0;
    double two_norm_error_sum_imag = 0.0;
    for (int i = 0; i < n*n_batch; i++)
    {
        double e1 = AbsError(tested[2 * i], standard[2 * i]);
        double e2 = AbsError(tested[2 * i + 1], standard[2 * i + 1]);
        epsc_error_sum_real += e1 * e1;
        epsc_error_sum_imag += e2 * e2;
        // printf("%lf %lf\n",epsc_error_sum_real, epsc_error_sum_imag);

        double e3 = standard[2 * i];
        double e4 = standard[2 * i + 1];
        two_norm_error_sum_real += e3 * e3;
        two_norm_error_sum_imag += e4 * e4;
    }

    real_error = sqrt(epsc_error_sum_real) / sqrt(two_norm_error_sum_real);
    imag_error = sqrt(epsc_error_sum_imag) / sqrt(two_norm_error_sum_imag);

    if (real_error < 1e-5)
    {
        std::cout << "The FFT computation is correct, real error is " << real_error << ", imag error is " << imag_error << std::endl;
        return 1;
    }
    else
    {
        std::cout << "The FFT computation is not correct, real error is " << real_error << ", imag error is " << imag_error << std::endl;
        return 0;
    }
}

int main(int argc, char *argv[])
{
    int n = 65536, n_batch = 1, seed = 42;
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "n:b:s:")))
    {
        switch (opt_c)
        {
        case 'n':
            n = atoi(optarg);
            break;
        case 'b':
            n_batch = atoi(optarg);
            break;
        case 's':
            seed = atoi(optarg);
            break;
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }



    std::cout<<"n: " << n << ", n_batch: " << n_batch << ", seed: " << seed << std::endl;
    double *data = (double *)malloc(sizeof(double) * n * n_batch * 2);
    generate_data(data, n, n_batch, seed);

    double *standard = (double *)malloc(sizeof(double) * n * n_batch * 2);
    fftw3_get_result(data, standard, n, n_batch);


    double *tested = (double *)malloc(sizeof(double) * n * n_batch * 2);
    setup(data, n, n_batch);
    doit(1);
    finalize(tested);

    double max_abs_error = -1;
    double max_rel_error = -1;
    double correct_rate = -1;
    double real_error = -1;
    double imag_error = -1;

    correct_rate = get_error(tested, standard, n, n_batch, max_abs_error, max_rel_error);
    CompareEpscInterleave(tested, standard, n, n_batch, real_error, imag_error);

    printf("correct rate is %lf\n", correct_rate);
    printf("max abs error is %lf, max relative error is %lf\n", max_abs_error, max_rel_error);
    printf("\n\n");
    return 0;
}

//对于长度为512的数据，在输入gpu前在主存中实际的数据排列顺序为
// 0 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 
// 2 34 66 98 130 162 194 226 258 290 322 354 386 418 450 482
// 4 36 68 100 132 164 196 228 260 292 324 356 388 420 452 484 
// 6 38 70 102 134 166 198 230 262 294 326 358 390 422 454 486 
// 8 40 72 104 136 168 200 232 264 296 328 360 392 424 456 488 
// 10 42 74 106 138 170 202 234 266 298 330 362 394 426 458 490 
// 12 44 76 108 140 172 204 236 268 300 332 364 396 428 460 492 
// 14 46 78 110 142 174 206 238 270 302 334 366 398 430 462 494 
// 16 48 80 112 144 176 208 240 272 304 336 368 400 432 464 496 
// 18 50 82 114 146 178 210 242 274 306 338 370 402 434 466 498 
// 20 52 84 116 148 180 212 244 276 308 340 372 404 436 468 500 
// 22 54 86 118 150 182 214 246 278 310 342 374 406 438 470 502 
// 24 56 88 120 152 184 216 248 280 312 344 376 408 440 472 504 
// 26 58 90 122 154 186 218 250 282 314 346 378 410 442 474 506 
// 28 60 92 124 156 188 220 252 284 316 348 380 412 444 476 508 
// 30 62 94 126 158 190 222 254 286 318 350 382 414 446 478 510 
// 1 33 65 97 129 161 193 225 257 289 321 353 385 417 449 481 
// 3 35 67 99 131 163 195 227 259 291 323 355 387 419 451 483 
// 5 37 69 101 133 165 197 229 261 293 325 357 389 421 453 485 
// 7 39 71 103 135 167 199 231 263 295 327 359 391 423 455 487 
// 9 41 73 105 137 169 201 233 265 297 329 361 393 425 457 489 
// 11 43 75 107 139 171 203 235 267 299 331 363 395 427 459 491 
// 13 45 77 109 141 173 205 237 269 301 333 365 397 429 461 493 
// 15 47 79 111 143 175 207 239 271 303 335 367 399 431 463 495 
// 17 49 81 113 145 177 209 241 273 305 337 369 401 433 465 497 
// 19 51 83 115 147 179 211 243 275 307 339 371 403 435 467 499 
// 21 53 85 117 149 181 213 245 277 309 341 373 405 437 469 501 
// 23 55 87 119 151 183 215 247 279 311 343 375 407 439 471 503 
// 25 57 89 121 153 185 217 249 281 313 345 377 409 441 473 505 
// 27 59 91 123 155 187 219 251 283 315 347 379 411 443 475 507 
// 29 61 93 125 157 189 221 253 285 317 349 381 413 445 477 509 
// 31 63 95 127 159 191 223 255 287 319 351 383 415 447 479 511 
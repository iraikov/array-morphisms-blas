;;; tests/test-blas-egg-backend.scm
;;; Test suite for array-morphisms-blas-egg-backend (the system-BLAS adapter
;;; now packaged as its own egg, array-morphisms-blas, separate from the
;;; core array-morphisms egg so the 'blas' egg dependency stays optional).
;;;
;;; Mirrors the structure of array-morphisms/tests/test-microblas.scm.

(import scheme (chicken base))
(import test)
(import (only srfi-1 iota every))
(import srfi-4)
(import array-morphisms-blas-exec)
(import array-morphisms-realization)
(import array-morphisms-blas-egg-backend)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (approx= a b #!optional (tol 1e-6))
  (< (abs (- a b)) tol))

(define (rel-close? a b tol)
  (<= (abs (- a b)) (* tol (max 1.0 (abs a) (abs b)))))

(define (gen-a i) (sin (* (+ i 1) 0.123)))
(define (gen-b i) (cos (* (+ i 1) 0.071)))

(define (fill-f32vec! v n f)
  (do ((i 0 (+ i 1))) ((= i n) v)
    (f32vector-set! v i (exact->inexact (f i)))))

(define (naive-gemm M N K alpha A B beta C)
  (let ((out (make-vector (* M N) 0.0)))
    (do ((i 0 (+ i 1))) ((= i M) out)
      (do ((j 0 (+ j 1))) ((= j N))
        (let ((sum 0.0))
          (do ((k 0 (+ k 1))) ((= k K))
            (set! sum (+ sum (* (vector-ref A (+ (* i K) k))
                                 (vector-ref B (+ (* k N) j))))))
          (vector-set! out (+ (* i N) j)
                       (+ (* alpha sum) (* beta (vector-ref C (+ (* i N) j))))))))))

(define (vec->f32 v)
  (let* ((n (vector-length v)) (out (make-f32vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) out)
      (f32vector-set! out i (exact->inexact (vector-ref v i))))))

(define (f32->vec v)
  (let* ((n (f32vector-length v)) (out (make-vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) out)
      (vector-set! out i (f32vector-ref v i)))))

(define (vec-close? a b tol)
  (and (= (vector-length a) (vector-length b))
       (let loop ((i 0))
         (or (= i (vector-length a))
             (and (rel-close? (vector-ref a i) (vector-ref b i) tol)
                  (loop (+ i 1)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1: Backend construction
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "blas-egg-backend - Backend construction"

  (test-assert "make-blas-egg-backend returns a blas-backend record"
    (blas-backend? (make-blas-egg-backend)))

  (test-assert "make-blas-egg-backend names the backend 'chicken-blas-egg"
    (eq? 'chicken-blas-egg (blas-backend-name (make-blas-egg-backend))))

  (test-assert "all 16 kernel slots are populated (non-#f)"
    (let ((b (make-blas-egg-backend)))
      (every (lambda (x) x)
             (list (blas-backend-gemm-f64 b) (blas-backend-gemm-f32 b)
                   (blas-backend-gemm-strided-f64 b) (blas-backend-gemm-strided-f32 b)
                   (blas-backend-gemv-f64 b) (blas-backend-gemv-f32 b)
                   (blas-backend-dot-f64 b) (blas-backend-dot-f32 b)
                   (blas-backend-axpy-f64 b) (blas-backend-axpy-f32 b)
                   (blas-backend-conv-fwd-im2col-f32 b)
                   (blas-backend-conv-bwd-data-im2col-f32 b)
                   (blas-backend-conv-bwd-weights-im2col-f32 b)
                   (blas-backend-conv-fwd-nhwc-im2col-f32 b)
                   (blas-backend-conv-bwd-data-nhwc-im2col-f32 b)
                   (blas-backend-conv-bwd-weights-nhwc-im2col-f32 b)))))

  (test-assert "register-blas-backend! + blas-available? round-trip"
    (let ((saved *active-backend*))
      (register-blas-backend! (make-blas-egg-backend))
      (let ((r (and (blas-available?)
                    (eq? 'chicken-blas-egg (blas-backend-name (active-blas-backend))))))
        (set! *active-backend* saved)
        r)))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2: GEMM correctness (plain, contiguous)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "blas-egg-backend - GEMM correctness"

  (let* ((be (make-blas-egg-backend))
         (gemm-f32 (blas-backend-gemm-f32 be)))
    (for-each
     (lambda (size)
       (let* ((M size) (N size) (K size))
         (test-assert (string-append "gemm-f32 square size=" (number->string size))
           (let* ((A (make-f32vector (* M K) 0.0)) (Av (make-vector (* M K) 0.0))
                  (B (make-f32vector (* K N) 0.0)) (Bv (make-vector (* K N) 0.0))
                  (C (make-f32vector (* M N) 0.0)))
             (do ((i 0 (+ i 1))) ((= i (* M K)))
               (f32vector-set! A i (exact->inexact (gen-a i)))
               (vector-set! Av i (exact->inexact (gen-a i))))
             (do ((i 0 (+ i 1))) ((= i (* K N)))
               (f32vector-set! B i (exact->inexact (gen-b i)))
               (vector-set! Bv i (exact->inexact (gen-b i))))
             (gemm-f32 M N K 1.0 A B 0.0 C)
             (vec-close? (f32->vec C) (naive-gemm M N K 1.0 Av Bv 0.0 (make-vector (* M N) 0.0)) 1e-3)))))
     '(1 2 8 33 64))
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3: GEMM-strided correctness (transposed operands)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "blas-egg-backend - GEMM-strided correctness"

  (let* ((be (make-blas-egg-backend))
         (gemm-s-f32 (blas-backend-gemm-strided-f32 be)))

    (test-assert "strided: G x B^T (var-matmul dA backward pattern)"
      ;; G = [[1,2],[4,5]], B = [[1,0],[0,1],[1,1]], B^T = [[1,0,1],[0,1,1]]
      ;; G x B^T = [[1,2,3],[4,5,9]]
      (let* ((G (vec->f32 #(1.0 2.0 4.0 5.0)))
             (B (vec->f32 #(1.0 0.0 0.0 1.0 1.0 1.0)))  ; physical 3x2
             (Cbuf (make-f32vector 6 0.0)))
        (gemm-s-f32 2 3 2 1.0 G 2 'no-trans B 2 'trans 0.0 Cbuf)
        (vec-close? (f32->vec Cbuf) #(1.0 2.0 3.0 4.0 5.0 9.0) 1e-3)))
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4: DOT / AXPY correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "blas-egg-backend - DOT / AXPY correctness"

  (let* ((be (make-blas-egg-backend))
         (dot-f32 (blas-backend-dot-f32 be))
         (axpy-f32 (blas-backend-axpy-f32 be)))

    (test-assert "dot-f32 basic"
      (approx= (dot-f32 3 (vec->f32 #(1.0 2.0 3.0)) (vec->f32 #(4.0 5.0 6.0))) 32.0 1e-3))

    (test-assert "axpy-f32 basic: y := 2*x + y"
      (let ((x (vec->f32 #(1.0 2.0 3.0)))
            (y (vec->f32 #(10.0 10.0 10.0))))
        (axpy-f32 3 2.0 x y)
        (vec-close? (f32->vec y) #(12.0 14.0 16.0) 1e-3)))
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5: Conv im2col hot-kernel hooks (NCHW and NHWC)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "blas-egg-backend - Conv im2col hot-kernel hooks"

  (let ((saved *active-backend*))
    (register-blas-backend! (make-blas-egg-backend))

    (let* ((N 2) (C 2) (H 5) (W 5) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
           (OH 5) (OW 5) (out-ch 3)
           (fan-in (* C KH KW)) (M (* N OH OW))
           (x-shape (vector N C H W))
           (x-shape-nhwc (vector N H W C))
           (src (make-f32vector (* N C H W) 0.0))
           (src-nhwc (make-f32vector (* N H W C) 0.0))
           (wt  (make-f32vector (* fan-in out-ch) 0.0))
           (b   (make-f32vector out-ch 0.0))
           (g   (make-f32vector (* M out-ch) 0.0)))
      (fill-f32vec! src (* N C H W) gen-a)
      (fill-f32vec! src-nhwc (* N H W C) gen-a)
      (fill-f32vec! wt (* fan-in out-ch) gen-b)
      (fill-f32vec! b out-ch (lambda (i) (* 0.1 (+ i 1))))
      (fill-f32vec! g (* M out-ch) (lambda (i) (sin (* (+ i 3) 0.211))))

      (test-assert "conv-fwd (NCHW): hook matches scalar Scheme reference"
        (let ((ref-out (make-f32vector (* M out-ch) 0.0))
              (blas-out (make-f32vector (* M out-ch) 0.0)))
          (execute-conv-fwd-nchw ref-out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-fwd-blas blas-out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-out) (f32->vec ref-out) 1e-2)))

      (test-assert "conv-fwd (NHWC): hook matches scalar Scheme reference"
        (let ((ref-out (make-f32vector (* M out-ch) 0.0))
              (blas-out (make-f32vector (* M out-ch) 0.0)))
          (execute-conv-fwd-nhwc ref-out src-nhwc wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-fwd-nhwc-blas blas-out src-nhwc wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-out) (f32->vec ref-out) 1e-2)))

      (test-assert "conv-bwd-data (NCHW): hook matches scalar Scheme reference"
        (let ((ref-dx (make-f32vector (* N C H W) 0.0))
              (blas-dx (make-f32vector (* N C H W) 0.0)))
          (execute-conv-bwd-data-nchw ref-dx x-shape g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-data-blas blas-dx x-shape g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-dx) (f32->vec ref-dx) 1e-2)))

      (test-assert "conv-bwd-data (NHWC): hook matches scalar Scheme reference"
        (let ((ref-dx (make-f32vector (* N H W C) 0.0))
              (blas-dx (make-f32vector (* N H W C) 0.0)))
          (execute-conv-bwd-data-nhwc ref-dx x-shape-nhwc g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-data-nhwc-blas blas-dx x-shape-nhwc g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-dx) (f32->vec ref-dx) 1e-2)))

      (test-assert "conv-bwd-weights (NCHW): hook matches scalar Scheme reference"
        (let ((ref-dwt (make-f32vector (* fan-in out-ch) 0.0))
              (blas-dwt (make-f32vector (* fan-in out-ch) 0.0))
              (wt-shape (vector fan-in out-ch)))
          (execute-conv-bwd-weights-nchw ref-dwt wt-shape g #f src N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-weights-blas blas-dwt wt-shape g #f src N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-dwt) (f32->vec ref-dwt) 1e-2)))

      (test-assert "conv-bwd-weights (NHWC): hook matches scalar Scheme reference"
        (let ((ref-dwt (make-f32vector (* fan-in out-ch) 0.0))
              (blas-dwt (make-f32vector (* fan-in out-ch) 0.0))
              (wt-shape (vector fan-in out-ch)))
          (execute-conv-bwd-weights-nhwc ref-dwt wt-shape g #f src-nhwc N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-weights-nhwc-blas blas-dwt wt-shape g #f src-nhwc N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-dwt) (f32->vec ref-dwt) 1e-2))))

    (set! *active-backend* saved))
)

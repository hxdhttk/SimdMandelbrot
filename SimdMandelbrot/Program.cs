using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace SimdMandelbrot
{
    using Vec = Vector256<double>;

    class Program
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static Vec VecInit(double value) => Vector256.CreateScalar(value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool VecIsAnyLe(Vec v, Vec f)
        {
            var m = Avx2.Compare(v, f, FloatComparisonMode.OrderedLessThanOrEqualSignaling);
            return !Avx2.TestZ(m, m);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int VecIsLe(Vec v1, Vec v2) => Avx2.MoveMask(Avx2.Compare(v1, v2, FloatComparisonMode.OrderedLessThanOrEqualSignaling));

        static readonly byte[] _bitRev =
        {
            0b0000, 0b1000, 0b0100, 0b1100, 0b0010, 0b1010, 0b0110, 0b1110,
            0b0001, 0b1001, 0b0101, 0b1101, 0b0011, 0b1011, 0b0111, 0b1111
        };

        static readonly int _vecSize = Unsafe.SizeOf<Vec>() / sizeof(double);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool VecAllNle(in Vec[] v1, Vec v2)
        {
            for (var i = 0; i < 8 / _vecSize; i++)
            {
                if (VecIsAnyLe(v1[i], v2))
                {
                    return false;
                }
            }

            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static uint Pixels(in Vec[] value, Vec limit)
        {
            uint res = 0;
            for (var i = 0; i < 8 / _vecSize; i++)
            {
                res <<= _vecSize;
                res |= _bitRev[VecIsLe(value[i], limit)];
            }

            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void CalcSum(ref Vec[] real, ref Vec[] imag, ref Vec[] sum, in Vec[] initReal, Vec initImag)
        {
            for (var i = 0; i < 8 / _vecSize; i++)
            {
                var r2 = Avx2.Multiply(real[i], real[i]);
                var i2 = Avx2.Multiply(imag[i], imag[i]);
                var ri = Avx2.Multiply(real[i], imag[i]);

                sum[i] = Avx2.Add(r2, i2);

                real[i] = Avx2.Add(Avx2.Subtract(r2, i2), initReal[i]);
                imag[i] = Avx2.Add(Avx2.Add(ri, ri), initImag);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static uint Mand8(bool toPrune, in Vec[] initReal, Vec initImag)
        {
            var vec40 = VecInit(4.0);
            var real = new Vec[8 / _vecSize];
            var imag = new Vec[8 / _vecSize];
            var sum = new Vec[8 / _vecSize];

            for (var k = 0; k < 8 / _vecSize; k++)
            {
                real[k] = initReal[k];
                imag[k] = initImag;
            }

            if (toPrune)
            {
                for (var j = 0; j < 12; j++)
                {
                    for (var k = 0; k < 4; k++)
                    {
                        CalcSum(ref real, ref imag, ref sum, initReal, initImag);
                    }

                    if (VecAllNle(sum, vec40))
                    {
                        return 0;
                    }
                }
            }
            else
            {
                for (var j = 0; j < 8; j++)
                {
                    for (var k = 0; k < 6; k++)
                    {
                        CalcSum(ref real, ref imag, ref sum, initReal, initImag);
                    }
                }

                CalcSum(ref real, ref imag, ref sum, initReal, initImag);
                CalcSum(ref real, ref imag, ref sum, initReal, initImag);
            }

            return Pixels(sum, vec40);
        }

        static void Main(string[] args)
        {
            var widHt = 16000;

            if (args.Length >= 1)
            {
                int.TryParse(args[0], out widHt);
            }

            widHt = -(-widHt & -8);
            var width = widHt;
            var height = widHt;

            var dataLength = height * (width >> 3);
            var pixels = new byte[dataLength];

            var vecCount = width / _vecSize;
            var r0 = new Vec[vecCount];

            for (var x = 0; x < width; x++)
            {
                var initVal = 2.0 / width * x - 1.5;

                var vecIndex = x / _vecSize;
                var valIndex = x % _vecSize;

                r0[vecIndex] = r0[vecIndex].WithElement(valIndex, initVal);
            }

            Parallel.For(0, height, y =>
            {
                var iy = 2.0 / height * y - 1.0;
                var initImag = VecInit(iy);
                var rowStart = y * width / 8;
                var toPrune = false;

                for (var x = 0; x < width; x += 8)
                {
                    var initReal = new Vec[8 / _vecSize];
                    for (var i = 0; i < initReal.Length; i++)
                    {
                        initReal[i] = r0[x / _vecSize + i];
                    }

                    var res = Mand8(toPrune, initReal, initImag);
                    pixels[rowStart + x / 8] = Convert.ToByte(res);
                    toPrune = !Convert.ToBoolean(res);
                }
            });

            Console.WriteLine($"P4\n{width} {height}");

            Console.OpenStandardOutput().Write(pixels, 0, dataLength);
        }
    }
}
